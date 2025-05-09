import re
from typing import Union, Optional
from .tree import Condition

class MetaRule:
    """表示由LLM生成的元规则，包含特征、分裂值和置信度"""
    
    def __init__(
        self, 
        feature_idx: int, 
        feature_name: str,
        split_value: Union[float, str, int], 
        is_categorical: bool, 
        confidence: int
    ):
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.split_value = split_value
        self.is_categorical = is_categorical
        self.confidence = confidence
    
    def to_condition(self) -> Condition:
        """转换为Condition对象"""
        if self.is_categorical:
            return Condition.categorical(self.feature_idx, {self.split_value})
        else:
            return Condition.numerical(self.feature_idx, None, self.split_value)
    
    def __str__(self) -> str:
        """返回规则的字符串表示"""
        op = "=" if self.is_categorical else "<"
        return f"{self.feature_name} {op} {self.split_value} [ confidence: {self.confidence} ]"
    
    @staticmethod
    def parse_rule(rule_str: str, meta) -> Optional['MetaRule']:
        """从规则字符串解析MetaRule"""
        # 提取特征名称、操作符、值和置信度
        pattern = r'(.*?)\s*([<>=]+)\s*([\d\.]+)\s*\[\s*confidence:\s*(\d+)\s*\]'
        match = re.match(pattern, rule_str.strip())
        
        if not match:
            return None
            
        feature_name, operator, value_str, confidence_str = match.groups()
        feature_name = feature_name.strip()
        value_str = value_str.strip()
        confidence = int(confidence_str)
        
        # 查找特征索引
        feature_idx = -1
        for idx, feature in enumerate(meta.features):
            if feature.name.lower() == feature_name.lower():
                feature_idx = idx
                break
        
        if feature_idx == -1:
            return None
            
        # 判断是分类特征还是数值特征
        feature = meta.features[feature_idx]
        is_categorical = operator == '='
        
        # 解析分裂值
        if is_categorical:
            split_value = value_str
        else:
            # 数值型特征
            if feature.type == 'int':
                split_value = int(float(value_str))
            else:
                split_value = float(value_str)
        
        return MetaRule(
            feature_idx=feature_idx,
            feature_name=feature_name,
            split_value=split_value,
            is_categorical=is_categorical,
            confidence=confidence
        ) 