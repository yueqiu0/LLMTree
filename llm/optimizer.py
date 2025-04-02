# llm/optimizer.py
import json
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import jinja2
from pathlib import Path
from tree_prompt.runner.runner import Runner
from tree_prompt.dataset import DatasetMeta
from tree_prompt.model.tree import DecisionTree

logger = logging.getLogger(__name__)

@dataclass
class LLMTreeResult:
    features: List[int]          # 使用的特征索引列表
    rules: List[str]             # IF-THEN规则列表
    leaf_classes: List[int]      # 叶节点允许的标签值
    confidence: float = 1.0      # LLM生成置信度

class LLMTreeOptimizer:
    def __init__(
        self,
        runner: Runner,
        template_dir: str = "template",
        max_retries: int = 3
    ):
        """
        Args:
            runner: LLM运行器
            template_dir: 模板目录路径(根据项目结构调整为'template')
            max_retries: LLM调用最大重试次数
        """
        self.runner = runner
        self.max_retries = max_retries
        self._init_templates(template_dir)

    def _init_templates(self, template_dir: str):
        # 获取当前脚本的绝对路径
        current_script_path = Path(__file__).resolve()
        
        # 计算项目根目录（假设optimizer.py位于llm目录下）
        project_root = current_script_path.parent.parent  # llm → project_root
        
        # 构建模板目录的绝对路径
        template_path = project_root / template_dir
        print(f"模板绝对路径: {template_path}")  # 调试输出
        
        # 验证路径是否存在
        if not template_path.is_dir():
            raise NotADirectoryError(f"模板目录不存在: {template_path}")
        
        # 初始化Jinja环境
        loader = jinja2.FileSystemLoader(searchpath=str(template_path))
        self.env = jinja2.Environment(loader=loader)
        
        # 加载模板
        try:
            self.tree_gen_template = self.env.get_template("zero_shot_tree.jinja")
            self.tree_compare_template = self.env.get_template("tree_comparison.jinja")
        except jinja2.TemplateNotFound as e:
            available = loader.list_templates()
            raise RuntimeError(
                f"模板 '{e.name}' 未找到，可用模板: {available}"
            ) from e

    def generate_tree(
        self,
        meta: DatasetMeta,
        examples: List,
        max_depth: int
    ) -> Optional[LLMTreeResult]:
        """生成LLM优化的决策树"""
        prompt = self._build_generation_prompt(meta, examples, max_depth)
        
        for attempt in range(self.max_retries):
            try:
                response = self._get_llm_response(prompt)
                return self._parse_response(response, meta)
            except Exception as e:
                logger.warning(f"LLM生成尝试 {attempt + 1} 失败: {str(e)}")
                
        logger.error("LLM树生成达到最大重试次数")
        return None

    def compare_trees(
        self,
        original_rules: List[str],
        llm_rules: List[str],
        criteria: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """对比两种决策树规则"""
        prompt = self._build_comparison_prompt(original_rules, llm_rules, criteria)
        
        try:
            response = self._get_llm_response(prompt)
            return self._parse_comparison(response)
        except Exception as e:
            logger.error(f"树对比失败: {str(e)}")
            return None

    # ------------------ 私有方法 ------------------
    def _build_generation_prompt(
        self,
        meta: DatasetMeta,
        examples: List,
        max_depth: int
    ) -> str:
        """构建树生成提示"""
        return self.tree_gen_template.render(
            features=meta.features,
            labels=meta.labels,
            examples=examples[:5],  # 限制示例数量
            max_depth=max_depth,
            feature_count=meta.feature_count()
        )

    def _build_comparison_prompt(
        self,
        original_rules: List[str],
        llm_rules: List[str],
        criteria: Optional[List[str]] = None
    ) -> str:
        """构建树对比提示"""
        default_criteria = [
            "特征选择合理性",
            "分裂阈值业务逻辑匹配度",
            "规则可解释性"
        ]
        return self.tree_compare_template.render(
            original_rules=original_rules[:50],  # 限制规则数量
            llm_rules=llm_rules[:50],
            criteria=criteria or default_criteria
        )

    def _get_llm_response(self, prompt: str) -> str:
        """执行LLM调用并验证响应"""
        responses = list(self.runner.run([prompt]))
        if not responses or not responses[0]:
            raise RuntimeError("未获得LLM响应")
        return responses[0][0].strip()

    def _parse_response(self, response: str, meta: DatasetMeta) -> LLMTreeResult:
        """解析LLM生成的树结构"""
        try:
            data = json.loads(self._extract_json(response))
            
            # 验证特征索引
            valid_features = [
                fid for fid in data["features"] 
                if 0 <= fid < meta.feature_count()
            ]
            
            # 转换标签值
            leaf_classes = []
            for name in data["leaf_classes"]:
                if val := meta.get_label_value(name):
                    leaf_classes.append(val)
                else:
                    raise ValueError(f"未知标签名称: {name}")
                    
            return LLMTreeResult(
                features=valid_features,
                rules=data["rules"],
                leaf_classes=leaf_classes,
                confidence=data.get("confidence", 1.0)
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}") from e

    def _parse_comparison(self, response: str) -> Dict:
        """解析对比结果"""
        try:
            return json.loads(self._extract_json(response))
        except json.JSONDecodeError as e:
            raise ValueError(f"对比结果解析失败: {str(e)}") from e

    @staticmethod
    def _extract_json(text: str) -> str:
        """从响应文本提取JSON内容"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("未找到JSON内容")
        return text[start:end]