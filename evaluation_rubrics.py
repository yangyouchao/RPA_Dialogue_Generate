"""Rolecraft criteria adapted for reference-free teacher response evaluation.

Source: rolecraft角色扮演评估/evaluation_rubrics.py (root version).
Keep its role/open-dialogue dimensions and 1/3/5 scale; replace comparative
anchors with absolute criteria. Do not map generation modes to its scenarios.
"""

ROLE_CONSISTENCY = {
    "dimension_id": "role_consistency",
    "dimension_name": "角色一致性",
    "intro": "判断待测回复是否忠实扮演目标角色，并适应当前情境。",
    "principles": {
        "性格特质": "模型的输出是否符合角色的核心人格特质。1=与角色核心人格特质明显冲突；3=基本符合，但角色特质体现笼统、薄弱或不稳定；5=自然、具体且稳定地体现角色核心人格特质。",
        "语言风格": "模型的输出是否符合角色的语言风格。1=明显不符合角色语言风格；3=基本符合，但表达较通用或机械堆砌标志性话语；5=自然、稳定地体现角色语言风格，且表达贴合本轮内容。",
        "情境适应": "模型的输出是否符合当前的情境。1=明显误解或违背上下文情境；3=基本符合，但对情境中的重要细节回应不足；5=准确理解当前情境，以符合角色的方式自然回应其中的重要细节。",
    },
}

OPEN_DIALOGUE = {
    "dimension_id": "open_domain_ability",
    "dimension_name": "开放域对话任务能力",
    "intro": "忽略角色化表达本身，只评价开放域对话能力与话题深度。",
    "principles": {
        "话题承接": "是否承接并推进话题。1=完全无法承接上文；3=能承接并回应；5=承接基础上给出有趣或有信息量的新角度，能吸引继续交流。",
        "话题深度": "完全剥离角色风格、比喻等表达手段，只评价话语内核对话题的理解与信息密度。1=理解肤浅或偏离话题，缺少有效信息；3=理解基本准确，提供与当前话题相关的有效信息；5=展现深入理解，提供具体、有信息量且有启发性的分析或见解。",
    },
}


def dimensions():
    """One call per principle, as in the source benchmark."""
    return [dict(group, principles={name: criterion})
            for group in (ROLE_CONSISTENCY, OPEN_DIALOGUE)
            for name, criterion in group["principles"].items()]
