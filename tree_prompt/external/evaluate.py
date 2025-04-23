def evaluate(tree_model, data):
    # 原始代码
    # tree_model.rules = tree_model.get_rules()
    
    # 修改后代码：增加对规则类型的检查
    rules = tree_model.get_rules()
    if isinstance(rules, str):
        print("Warning: Rules returned as a string. Converting to list of rules.")
        rules = [rules]  # 将字符串转换为规则列表
    tree_model.rules = rules

    # 原始代码
    # predictions = tree_model.predict(data)
    
    # 修改后代码：增加对规则类型的检查
    if not isinstance(tree_model.rules, list):
        raise TypeError("Rules must be a list of objects, not a string or other type.")
    predictions = tree_model.predict(data)
    return predictions