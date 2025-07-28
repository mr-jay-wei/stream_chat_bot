# rag/prompt_manager.py

import os
from pathlib import Path
from typing import Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


class PromptManager:
    """
    提示词管理器，负责加载和管理所有提示词模板。
    实现提示词与代码的解耦。
    """
    
    def __init__(self):
        """初始化提示词管理器。"""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._prompt_cache: Dict[str, str] = {}
        self._template_cache: Dict[str, PromptTemplate] = {}
        
        # 确保提示词目录存在
        self.prompts_dir.mkdir(exist_ok=True)
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        加载指定的提示词内容。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词内容字符串
            
        Raises:
            FileNotFoundError: 如果提示词文件不存在
        """
        # 检查缓存
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        
        # 构建文件路径
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
        
        # 读取文件内容
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 缓存内容
            self._prompt_cache[prompt_name] = content
            return content
            
        except Exception as e:
            raise RuntimeError(f"读取提示词文件失败 {prompt_file}: {e}")
    
    def get_template(self, prompt_name: str) -> PromptTemplate:
        """
        获取指定的提示词模板对象。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            LangChain PromptTemplate 对象
        """
        # 检查缓存
        if prompt_name in self._template_cache:
            return self._template_cache[prompt_name]
        
        # 加载提示词内容
        prompt_content = self.load_prompt(prompt_name)
        
        # 创建模板对象
        template = PromptTemplate.from_template(prompt_content)
        
        # 缓存模板
        self._template_cache[prompt_name] = template
        return template
    
    def reload_prompt(self, prompt_name: str) -> str:
        """
        重新加载指定的提示词（清除缓存后重新读取）。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词内容字符串
        """
        # 清除缓存
        self._prompt_cache.pop(prompt_name, None)
        self._template_cache.pop(prompt_name, None)
        
        # 重新加载
        return self.load_prompt(prompt_name)
    
    def list_available_prompts(self) -> list:
        """
        列出所有可用的提示词文件。
        
        Returns:
            提示词文件名列表（不含扩展名）
        """
        prompt_files = []
        # 使用 pathlib.Path.glob() 方法 (推荐)
        for file_path in self.prompts_dir.glob("*.txt"):
            prompt_files.append(file_path.stem)  # .stem 获取不含扩展名的文件名
        return sorted(prompt_files)
        
        # 如果使用标准库 glob 的等价写法：
        # import glob
        # pattern = str(self.prompts_dir / "*.txt")
        # for file_path in glob.glob(pattern):
        #     filename = os.path.splitext(os.path.basename(file_path))[0]
        #     prompt_files.append(filename)
    
    def save_prompt(self, prompt_name: str, content: str) -> None:
        """
        保存提示词到文件。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            content: 提示词内容
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            # 清除缓存，确保下次加载时使用新内容
            self._prompt_cache.pop(prompt_name, None)
            self._template_cache.pop(prompt_name, None)
            
            print(f"提示词已保存到: {prompt_file}")
            
        except Exception as e:
            raise RuntimeError(f"保存提示词文件失败 {prompt_file}: {e}")
    
    def clear_cache(self) -> None:
        """清除所有缓存。"""
        self._prompt_cache.clear()
        self._template_cache.clear()
        print("提示词缓存已清除")
    
    def reload_all_prompts(self) -> Dict[str, str]:
        """
        重新加载所有提示词。
        
        Returns:
            重新加载的提示词字典
        """
        # 清除所有缓存
        self.clear_cache()
        
        # 重新加载所有提示词
        reloaded_prompts = {}
        for prompt_name in self.list_available_prompts():
            try:
                content = self.load_prompt(prompt_name)
                reloaded_prompts[prompt_name] = content
                print(f"✅ 重新加载: {prompt_name}")
            except Exception as e:
                print(f"❌ 重新加载失败 {prompt_name}: {e}")
        
        return reloaded_prompts
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """
        获取提示词的详细信息。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            提示词信息字典
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            return {"exists": False, "error": f"提示词文件不存在: {prompt_file}"}
        
        try:
            stat = prompt_file.stat()
            content = self.load_prompt(prompt_name)
            template = self.get_template(prompt_name)
            
            return {
                "exists": True,
                "file_path": str(prompt_file),
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "content_length": len(content),
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "template_variables": template.input_variables,
                "is_cached": prompt_name in self._prompt_cache
            }
        except Exception as e:
            return {"exists": True, "error": f"获取提示词信息失败: {e}"}
    
    def validate_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        验证提示词模板的有效性。
        
        Args:
            prompt_name: 提示词文件名（不含扩展名）
            
        Returns:
            验证结果字典
        """
        try:
            template = self.get_template(prompt_name)
            
            # 检查必需的变量
            required_vars = {"context", "question"}  # 问答提示词的必需变量
            missing_vars = required_vars - set(template.input_variables)
            extra_vars = set(template.input_variables) - required_vars
            
            # 尝试格式化测试
            test_values = {var: f"test_{var}" for var in template.input_variables}
            try:
                formatted = template.format(**test_values)
                format_test = {"success": True, "formatted_length": len(formatted)}
            except Exception as e:
                format_test = {"success": False, "error": str(e)}
            
            return {
                "valid": len(missing_vars) == 0 and format_test["success"],
                "template_variables": template.input_variables,
                "missing_variables": list(missing_vars),
                "extra_variables": list(extra_vars),
                "format_test": format_test
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"验证提示词失败: {e}"
            }


# 创建全局提示词管理器实例
prompt_manager = PromptManager()

'''
def get_qa_prompt_template() -> PromptTemplate:
    """获取问答提示词模板。"""
    return prompt_manager.get_template("qa_prompt")


def get_query_rewrite_prompt_template() -> PromptTemplate:
    """获取问题改写提示词模板。"""
    return prompt_manager.get_template("query_rewrite_prompt")


def load_qa_prompt() -> str:
    """加载问答提示词内容。"""
    return prompt_manager.load_prompt("qa_prompt")


def load_query_rewrite_prompt() -> str:
    """加载问题改写提示词内容。"""
    return prompt_manager.load_prompt("query_rewrite_prompt")
'''