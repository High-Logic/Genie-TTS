from typing import List, Tuple, Set


class ErhuaProcessor:
    """
    处理中文G2P中的儿化音逻辑。
    """

    def __init__(self):
        self.must_erhua: Set[str] = {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        }
        self.not_erhua: Set[str] = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
            "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
            "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿",
            "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿",
            "猪儿", "猫儿", "狗儿", "少儿",
        }

    def merge_erhua(self, initials: List[str], finals: List[str], word: str, pos: str) -> Tuple[List[str], List[str]]:
        # 1. 修正 er1 发音为 er2 (当'儿'在词尾且发音为er1时)
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == "er1":
                finals[i] = "er2"
        # 2. 检查是否跳过儿化处理
        if word not in self.must_erhua and (word in self.not_erhua or pos in {"a", "j", "nr"}):
            return initials, finals
        # 3. 长度校验 (处理如 "……" 等长度不一致的特殊符号情况)
        if len(finals) != len(word):
            return initials, finals
        # 4. 执行儿化合并逻辑 (与前一个字发同音)
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            # 判断是否需要合并儿化音
            # 条件: 是最后一个字 + 是"儿" + 发音是er2/er5 + 后两字不在非儿化表中 + 前面已有韵母
            if (
                    i == len(finals) - 1
                    and word[i] == "儿"
                    and phn in {"er2", "er5"}
                    and word[-2:] not in self.not_erhua
                    and new_finals
            ):
                # 将 'er' 加上前一个字的声调
                phn = "er" + new_finals[-1][-1]
            new_initials.append(initials[i])
            new_finals.append(phn)
        return new_initials, new_finals


if __name__ == "__main__":
    processor = ErhuaProcessor()
    print("===== 儿化逻辑测试 =====")
    test_cases = [
        # (initials, finals, word, pos, description)
        (["h", ""], ["hua1", "er2"], "花儿", "n", "普通儿化: 花儿 -> hua1 er1"),
        (["n", ""], ["nv3", "er2"], "女儿", "n", "非儿化名单: 女儿 -> 保持原样"),
        (["x", "y", ""], ["xiao3", "yuan4", "er2"], "小院儿", "n", "必儿化名单: 小院儿 -> er4"),
        (["w", ""], ["wan2", "er1"], "玩儿", "v", "修正er1 -> er2 -> er2 (动词通常儿化)"),
    ]
    for _initials, _finals, _word, _pos, _desc in test_cases:
        new_ini, new_fin = processor.merge_erhua(list(_initials), list(_finals), _word, _pos)
        print(f"测试: {_word} ({_desc})")
        print(f"  原始韵母: {_finals}")
        print(f"  结果韵母: {new_fin}")
        print("-" * 30)
