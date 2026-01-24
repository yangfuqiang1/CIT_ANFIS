import json
import xgboost as xgb


def parse_xgb_rules(xgb_model):
    """
    解析 XGBoost 模型，提取每一棵树的每一条路径作为一条规则。
    返回:
        rules: 列表，每个元素是一个规则对象
        {
            'conditions': [(feature_idx, sign, threshold), ...],
            'leaf_value': float
        }
    """
    # 获取 XGBoost 的 JSON 结构
    dump_list = xgb_model.get_booster().get_dump(dump_format='json')

    all_rules = []

    for tree_json in dump_list:
        tree_structure = json.loads(tree_json)
        # 递归提取这棵树的所有规则
        tree_rules = _recurse_tree(tree_structure, current_conditions=[])
        all_rules.extend(tree_rules)

    print(f"[XG-ANFIS] Successfully extracted {len(all_rules)} rules from XGBoost.")
    return all_rules


def _recurse_tree(node, current_conditions):
    """
    递归遍历树节点，收集路径
    """
    # --- 1. 到达叶子节点 (Leaf) ---
    if 'leaf' in node:
        # 这条路径结束，生成一条规则
        return [{
            'conditions': current_conditions,
            'leaf_value': node['leaf']
        }]

    # --- 2. 分裂节点 (Split) ---
    # XGBoost 的 split_condition 通常是 < threshold
    split_feat_idx = int(node['split'][1:])  # 例如 "f5" -> 5
    threshold = node['split_condition']

    rules = []

    if 'children' in node:
        yes_id = node['yes']  # 满足条件 (x < threshold)
        no_id = node['no']  # 不满足条件 (x >= threshold)

        # 找到对应的子节点对象
        left_node = next(n for n in node['children'] if n['nodeid'] == yes_id)
        right_node = next(n for n in node['children'] if n['nodeid'] == no_id)

        # 左分支规则 (x < threshold) -> sign = -1
        # Sigmoid( beta * (threshold - x) ) -> 当 x < thresh 时为正，激活
        cond_left = current_conditions + [(split_feat_idx, -1.0, threshold)]
        rules.extend(_recurse_tree(left_node, cond_left))

        # 右分支规则 (x >= threshold) -> sign = 1
        # Sigmoid( beta * (x - threshold) ) -> 当 x > thresh 时为正，激活
        cond_right = current_conditions + [(split_feat_idx, 1.0, threshold)]
        rules.extend(_recurse_tree(right_node, cond_right))

    return rules