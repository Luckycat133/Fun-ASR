import os
import sys
import time
import torch
import soundfile as sf
import re
import importlib.util

# Fix for a module-name collision: nagisa (a forced-aligner dependency) does a
# bare `import model` internally. When this script runs from the Fun-ASR root
# (which also contains a model.py), that import wrongly resolves to Fun-ASR's
# model.py. We push nagisa's own package directory ahead of sys.path[0] so the
# bare import resolves to nagisa's own model.py. find_spec locates the package
# without executing its (broken) __init__, avoiding the collision entirely.
_nagisa_spec = importlib.util.find_spec("nagisa")
if _nagisa_spec is not None and _nagisa_spec.submodule_search_locations:
    _nagisa_pkg = _nagisa_spec.submodule_search_locations[0]
    if _nagisa_pkg not in sys.path:
        sys.path.insert(0, _nagisa_pkg)

from qwen_asr import Qwen3ASRModel

def align_punctuated_text_with_timestamps(transcribed_text, align_items):
    """
    将带标点的 transcribed_text 映射到无标点字符组成的 align_items 上。
    利用原始字符索引映射和动态同步恢复机制，完美避免中英文标点及大小写造成的对齐漂移。
    """
    nopunc_orig = []
    orig_indices = []
    for idx, char in enumerate(transcribed_text):
        if char.isalnum():
            nopunc_orig.append(char)
            orig_indices.append(idx)
    nopunc_str = "".join(nopunc_orig)

    mapped_items = []
    nopunc_idx = 0

    for item_idx, item in enumerate(align_items):
        item_text = item.text.replace(' ', '')
        if not item_text:
            continue
            
        match_len = len(item_text)
        sub = nopunc_str[nopunc_idx:nopunc_idx+match_len]
        
        if sub == item_text:
            start_nopunc = nopunc_idx
            end_nopunc = nopunc_idx + match_len
            
            start_orig = orig_indices[start_nopunc]
            end_orig = orig_indices[end_nopunc - 1]
            
            # 获取下一个词的起始偏移，以抓取中间夹带的所有标点符号
            next_start_orig = len(transcribed_text)
            if item_idx + 1 < len(align_items):
                next_item_text = align_items[item_idx + 1].text.replace(' ', '')
                if next_item_text:
                    next_start_nopunc = end_nopunc
                    if next_start_nopunc < len(orig_indices):
                        next_start_orig = orig_indices[next_start_nopunc]
                        
            recon_text = transcribed_text[start_orig:end_orig + 1]
            punc = transcribed_text[end_orig + 1:next_start_orig]
            
            mapped_items.append({
                'text': recon_text,
                'start_time': item.start_time,
                'end_time': item.end_time,
                'punctuation': punc
            })
            nopunc_idx = end_nopunc
        else:
            # 动态同步恢复：向后滑动寻找匹配
            found = False
            for offset in range(1, 20):
                if nopunc_idx + offset + match_len <= len(nopunc_str):
                    if nopunc_str[nopunc_idx+offset:nopunc_idx+offset+match_len] == item_text:
                        nopunc_idx += offset
                        found = True
                        break
            if found:
                start_nopunc = nopunc_idx
                end_nopunc = nopunc_idx + match_len
                start_orig = orig_indices[start_nopunc]
                end_orig = orig_indices[end_nopunc - 1]
                
                next_start_orig = len(transcribed_text)
                if item_idx + 1 < len(align_items):
                    next_item_text = align_items[item_idx + 1].text.replace(' ', '')
                    if next_item_text:
                        next_start_nopunc = end_nopunc
                        if next_start_nopunc < len(orig_indices):
                            next_start_orig = orig_indices[next_start_nopunc]
                            
                recon_text = transcribed_text[start_orig:end_orig + 1]
                punc = transcribed_text[end_orig + 1:next_start_orig]
                
                mapped_items.append({
                    'text': recon_text,
                    'start_time': item.start_time,
                    'end_time': item.end_time,
                    'punctuation': punc
                })
                nopunc_idx = end_nopunc
            else:
                # 强容错：如果恢复失败，直接强制记录当前词，不滑动 nopunc_idx
                mapped_items.append({
                    'text': item.text,
                    'start_time': item.start_time,
                    'end_time': item.end_time,
                    'punctuation': ''
                })
                
    return mapped_items

def mapped_items_to_srt(mapped_items, start_line_idx=1, max_chars=18, max_pause=0.8, replacements=None):
    """
    根据映射结果，合并字符生成 SRT 行，并在此过程中应用文本替换纠错。
    """
    srt_blocks = []
    current_line = []
    current_chars = 0
    line_start_time = None
    line_end_time = None
    
    line_counter = start_line_idx
    
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        
    def flush_line():
        nonlocal line_counter
        text_content = "".join(current_line).strip()
        if text_content:
            # 1. 自动去重语气词
            text_content = re.sub(r'^(呃|啊|哎|哦|嗯|哇|哈|对对对|行吧|好啦|啦|吧|呀|嘛|呢|呗|吧)+', '', text_content)
            text_content = re.sub(r'(呃|啊|哎|哦|嗯|哇|哈|好啦|啦|吧|呀|嘛|呢|呗|吧)+$', '', text_content)
            
            # 2. 叠字去重 (如 "我我" -> "我"，排除 "谢谢" "爸爸" 等)
            exclude_doubles = {"爸爸", "妈妈", "爷爷", "奶奶", "刚刚", "谢谢", "明明", "看看", "天天"}
            def deduplicate_chars(match):
                char = match.group(1)
                double = char + char
                return double if double in exclude_doubles else char
            text_content = re.sub(r'([\u4e00-\u9fa5])\1+', deduplicate_chars, text_content)
            
            # 3. 词组重叠去重
            for length in [3, 2]:
                pattern = rf'([\u4e00-\u9fa5]{{{length}}})\1'
                text_content = re.sub(pattern, r'\1', text_content)

            # 4. 应用专有名词纠错替换
            if replacements:
                for wrong, right in replacements.items():
                    text_content = re.sub(wrong, right, text_content)
            
            # 5. 清理因过滤语气词遗留的句首标点（如“，怎么熄火了” 变为 “怎么熄火了”）
            text_content = text_content.lstrip("，。？！、；：,.?!:; ")
            
            # 6. 清理字幕行尾无实际停顿的逗号或分隔符，保持字幕美观
            text_content = text_content.rstrip("，、, ")
            
            # 清理多余空格
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # 如果清理后没有文本了，就不输出此行
            if text_content:
                block = (
                    f"{line_counter}\n"
                    f"{format_time(line_start_time)} --> {format_time(line_end_time)}\n"
                    f"{text_content}\n"
                )
                srt_blocks.append(block)
                line_counter += 1
                
        current_line.clear()

    for it in mapped_items:
        word = it['text']
        punc = it['punctuation']
        start = it['start_time']
        end = it['end_time']
        
        full_word = word + punc
        
        if not current_line:
            current_line.append(full_word)
            current_chars = len(word)
            line_start_time = start
            line_end_time = end
        else:
            pause = start - line_end_time
            
            # 换行条件 1: 静音或字数超限
            if pause > max_pause or (current_chars + len(word)) > max_chars:
                flush_line()
                current_line.append(full_word)
                current_chars = len(word)
                line_start_time = start
                line_end_time = end
            else:
                current_line.append(full_word)
                current_chars += len(word)
                line_end_time = end
                
            # 换行条件 2: 遇到强标点 (。？！) 强制换行
            if any(p in punc for p in ["。", "？", "！", "!", "?", "；", ";"]):
                flush_line()
            # 换行条件 3: 遇到逗号类，且行长度达到 8 字以上，提前换行
            elif any(p in punc for p in ["，", "、", ","]) and current_chars >= 8:
                flush_line()
                
    if current_line:
        flush_line()
        
    return srt_blocks, line_counter

def main():
    # Audio / output paths can be overridden via CLI:
    #   python transcribe_yanhuo_to_srt_v2.py <audio.wav> <output.srt>
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/Shared/projects/yanhuo_audio.wav"
    output_srt_path = sys.argv[2] if len(sys.argv) > 2 else "/Users/Shared/projects/yanhuo_subtitle_final.srt"
    
    model_dir = "/Users/Shared/projects/Fun-ASR/models/Qwen3-ASR-1.7B"
    forced_aligner_dir = "/Users/Shared/projects/Fun-ASR/models/Qwen3-ForcedAligner-0.6B"

    replacements = {
        "新苗": "幸喵",
        "金彪": "幸喵",
        "金木": "幸喵",
        "王叔没的": "王叔们",
        "王烟火": "王叔",
        "Jan": "郑哥",
        "叶曼": "帮忙",
        "Zizi的Chat": "自己的车",
        "BABC": "BBC",
        "失火了": "熄火了",
        "车也嘛": "车怎么",
        "车改失火": "车给熄火",
        "车给失火": "车给熄火",
        "路队": "陆队",
        "要失败": "要死，被",
        "要失被": "要死，被",
        "吃坏了": "车坏了",
        "出坏了": "车坏了",
        "你才刚走来": "你才刚调来",
        "危害社交安全": "危害社会安全",
        "多土": "歹徒",
        "车坏了算么吧": "车坏了怎么算",
        "蓄意拍火": "蓄意纵火",
        "青潭镇": "青台镇",
        "林立修": "林理洵",
        "林黎新": "林理洵",
        "叶静山": "叶敬山",
    }

    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        sys.exit(1)

    print("正在加载 Qwen3-ASR-1.7B 及 Qwen3-ForcedAligner-0.6B...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    model = Qwen3ASRModel.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        forced_aligner=forced_aligner_dir,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=4,
        max_new_tokens=512
    )
    print("模型加载完成！")

    info = sf.info(audio_path)
    sr = info.samplerate
    total_frames = info.frames
    duration = info.duration
    
    chunk_sec = 900
    chunk_samples = chunk_sec * sr
    
    with open(output_srt_path, "w", encoding="utf-8") as f:
        pass

    current_frame = 0
    part_idx = 1
    srt_line_counter = 1
    start_time = time.time()

    with sf.SoundFile(audio_path) as audio_file:
        while current_frame < total_frames:
            frames_to_read = min(chunk_samples, total_frames - current_frame)
            if frames_to_read <= 0:
                break
                
            audio_data = audio_file.read(frames_to_read)
            chunk_start_sec = current_frame / sr
            chunk_end_sec = (current_frame + frames_to_read) / sr
            
            print(f"\n[分片 {part_idx}] 正在转录对齐 {chunk_start_sec:.1f}s 到 {chunk_end_sec:.1f}s ...")
            
            try:
                results = model.transcribe(
                    audio=(audio_data, sr),
                    language="Chinese",
                    return_time_stamps=True
                )
                
                transcription = results[0]
                raw_text = transcription.text
                align_res = transcription.time_stamps
                
                if align_res and align_res.items and raw_text:
                    # 1. 修正对齐时间戳全局偏移
                    global_items = []
                    for it in align_res.items:
                        global_items.append(
                            type(it)(
                                text=it.text,
                                start_time=round(it.start_time + chunk_start_sec, 3),
                                end_time=round(it.end_time + chunk_start_sec, 3)
                            )
                        )
                    
                    # 2. 正确做法：先以完全无错乱的原始 text 和 items 运行标点映射
                    mapped_items = align_punctuated_text_with_timestamps(raw_text, global_items)
                    
                    # 3. 将 mapped_items 合并成 SRT 字幕，并在此处安全执行纠错替换
                    srt_blocks, next_line_idx = mapped_items_to_srt(
                        mapped_items,
                        start_line_idx=srt_line_counter,
                        max_chars=18,
                        max_pause=0.8,
                        replacements=replacements
                    )
                    srt_line_counter = next_line_idx
                    
                    with open(output_srt_path, "a", encoding="utf-8") as f:
                        for block in srt_blocks:
                            f.write(block + "\n")
                        f.flush()
                        
                    print(f"[分片 {part_idx}]: 高精度语义对齐合并成功，生成了 {len(srt_blocks)} 行字幕。")
                else:
                    print(f"[分片 {part_idx}]: 未识别到有效语音。")
                    
            except Exception as e:
                print(f"转录分片 {part_idx} 时发生错误: {e}")
                
            current_frame += frames_to_read
            part_idx += 1
            
            # 显示进度
            elapsed = time.time() - start_time
            progress = current_frame / total_frames
            eta = (elapsed / progress) - elapsed if progress > 0 else 0
            print(f"进度: {progress*100:.2f}% | 已用时: {elapsed/60:.1f} 分钟 | 预计剩余: {eta/60:.1f} 分钟")
            
    print(f"\n高精度语义字幕生成完成！SRT 文件已保存至: {output_srt_path}")

if __name__ == "__main__":
    main()
