import re
# from tn.chinese.normalizer import Normalizer

from pypinyin import lazy_pinyin, Style
from pypinyin.core import load_phrases_dict

from text import pinyin_dict
from bert import TTSProsody


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_number(char):
    if char >= '0' and char <= '9':
        return True
    else:
        return False


numList = "零一二三四五六七八九"
digitList = ["", "十", "百", "千", "万"]
def number_to_chinese_low(number):  # 不能超过百万（不含百万）
    number = [int(s) for s in str(number)]
    numberLen = len(number)
    chinese = ""
    if numberLen == 1 and number[0] == 0:  # 如果是0
        chinese += numList[0]
    elif numberLen == 2 and number[0] == 1:  # 如果是十几
        chinese += digitList[1]
        if number[1]:  # 如果不是整十
            chinese += numList[number[1]]
    else:  # 非特殊情况
        for num in enumerate(number[::-1]):
            if not num[1] and len(chinese) > 0 and chinese[-1] != numList[0]:  # 多个0只读一个，低位无数字不读
                chinese += numList[0]
            elif num[1]:  # 不是零，正常读
                chinese += digitList[num[0]] + numList[num[1]]
        chinese = chinese[::-1]  # 反转，得到正常字符串
    return chinese


def number_to_chinese(number):  # 不能超过亿（不含亿）
    chinese = ""
    num0 = int(number // 10000)  # 获取高四位
    num1 = number % 10000  # 获取低四位
    if num0:  # 如果高位非零
        chinese += number_to_chinese_low(num0) + digitList[4]
        if num1 and num1 < 1000:  # 如果要添加零
            chinese += numList[0]
    if num1 or not num0:  # 如果低位非零或高位是零
        chinese += number_to_chinese_low(num1)
    return chinese

def number_to_chinese_test():
    print(number_to_chinese(10010910))
# number_to_chinese_test()

def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    numbers = []
    for char in text:
        if (is_chinese(char)):
            # 如果是数字，把数字转换为中文表达（优化）
            if len(numbers) > 0:
                if numbers[0] != '#':
                    numbers = number_to_chinese(int(''.join(numbers)))
                else:
                    numbers = ''.join([numList[int(num)] for num in numbers[1:]])
                text_clean.append(numbers)
                numbers = []
            text_clean.append(char)
        elif (is_number(char) or char == '#'):
            numbers.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean

def load_pinyin_dict():
    my_dict={}
    with open("./text/pinyin-local.txt", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            phone = cuts[1:]
            tmp = []
            for p in phone:
                tmp.append([p])
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


class VITS_PinYin:
    def __init__(self, bert_path, device, hasBert=True):
        load_pinyin_dict()
        self.hasBert = hasBert
        if self.hasBert:
            self.prosody = TTSProsody(bert_path, device)
        # self.normalizer = Normalizer()

    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    def chinese_to_phonemes(self, text):
        # text = self.normalizer.normalize(text)
        text = clean_chinese(text)
        phonemes = ["sil"]
        chars = ['[PAD]']
        count_phone = []
        count_phone.append(1)
        for subtext in text.split(","):
            if (len(subtext) == 0):
                continue
            pinyins = self.correct_pinyin_tone3(subtext)
            sub_p, sub_c = self.get_phoneme4pinyin(pinyins)
            phonemes.extend(sub_p)
            phonemes.append("sp")
            count_phone.extend(sub_c)
            count_phone.append(1)
            chars.append(subtext)
            chars.append(',')
        phonemes.append("sil")
        count_phone.append(1)
        chars.append('[PAD]')
        chars = "".join(chars)
        char_embeds = None
        if self.hasBert:
            char_embeds = self.prosody.get_char_embeds(chars)
            char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return " ".join(phonemes), char_embeds

    def correct_pinyin_tone3(self, text):
        pinyin_list = lazy_pinyin(text,
                                  style=Style.TONE3,
                                  strict=False,
                                  neutral_tone_with_five=True,
                                  tone_sandhi=True)
        # , tone_sandhi=True -> 33变调
        return pinyin_list

