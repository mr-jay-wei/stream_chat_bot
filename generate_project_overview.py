# generate_project_overview.py

import os
import fnmatch
from pathlib import Path
from typing import List, Set, Dict

# --- 可配置区域 ---

SCRIPT_NAME = "generate_project_overview.py"
OUTPUT_FILENAME = "generated_project_overview.md"

# 这是一个经过验证的、正确的忽略模式列表
IGNORE_PATTERNS: Set[str] = {
    ".git", ".vscode", ".idea", "__pycache__", "node_modules", "venv", ".venv",
    ".env", "build", "dist", "*.pyc", "*.egg-info", "*.log",
    # UV 和数据相关文件/文件夹
    "dist_electron",
    "dist_electron/*",
    "uv.lock", "data", "my_chromadb_vector_store", "*/.pytest_cache", 
    # 特定路径忽略
    "*/package-lock.json","package-lock.json","out/*",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.svg", "*.ico", "*.webp",
    "*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.m4a",
    "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.flv",
    "*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot",
    "*.zip", "*.rar", "*.7z", "*.tar", "*.gz",
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
    "*.exe", "*.dll", "*.so", "*.bin",
    # 忽略脚本自身和输出文件
    SCRIPT_NAME,
    OUTPUT_FILENAME,
}

# 语言映射
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".html": "html",
    ".css": "css", ".scss": "scss", ".json": "json", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".md": "text", ".sh": "shell",
    ".bat": "batch", ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby",
    ".swift": "swift", ".kt": "kotlin", "Dockerfile": "dockerfile",
}

# 需要特殊处理的文件类型
SPECIAL_HANDLING_EXTENSIONS = {".md", ".txt", ".rst"}

class ProjectDocumenter:
    """
    一个用于生成项目结构和文件内容文档的类。
    其核心忽略逻辑基于已验证的诊断脚本。
    """

    def __init__(self, root_dir: str, ignore_patterns: Set[str]):
        self.root_path = Path(root_dir).resolve()
        # 预先分离模式以优化性能
        self.plain_patterns = {p for p in ignore_patterns if "*" not in p}
        self.wildcard_patterns = ignore_patterns - self.plain_patterns
        print(f"项目根目录: {self.root_path}")

    def _should_ignore(self, path: Path) -> bool:
        """
        判断路径是否应该被忽略。
        规则：
        1. 普通忽略列表的目录/文件仍然忽略
        2. 通配符规则仍然生效
        3. 只有位于源码目录（electron、backend、frontend）下的 .cjs 文件才放行
        """
        try:
            relative_path = path.relative_to(self.root_path)
            relative_parts = relative_path.parts
            relative_path_str = relative_path.as_posix()
        except ValueError:
            return True  # 不在项目根目录下

        if not relative_parts:
            return False  # 根目录不忽略

        # 定义源码目录
        SOURCE_DIRS = {"electron", "backend", "frontend"}

        # 1. 检查完整路径是否在 plain_patterns 中
        if relative_path_str in self.plain_patterns:
            return True

        # 2. 检查路径任何部分是否在 plain_patterns 中
        if any(part in self.plain_patterns for part in relative_parts):
            # 放行源码目录下的 .cjs 文件
            if path.is_file() and path.suffix.lower() == ".cjs" and relative_parts[0] in SOURCE_DIRS:
                return False
            return True

        # 3. 检查文件名是否匹配通配符
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in self.wildcard_patterns):
            return True

        # 4. 检查完整路径是否匹配通配符
        if any(fnmatch.fnmatch(relative_path_str, pattern) for pattern in self.wildcard_patterns):
            return True

        # 5. 默认不过滤
        return False


    def get_filtered_paths(self) -> List[Path]:
        """
        获取项目中所有未被忽略的路径（文件和目录）。
        这是所有其他功能的数据来源，确保一致性。
        """
        filtered_paths = []
        
        def walk_directory(directory: Path):
            """递归遍历目录，跳过被忽略的目录"""
            try:
                for item in directory.iterdir():
                    if self._should_ignore(item):
                        continue
                    
                    filtered_paths.append(item)
                    
                    # 如果是目录且未被忽略，继续递归
                    if item.is_dir():
                        walk_directory(item)
            except (PermissionError, OSError):
                # 跳过无法访问的目录
                pass
        
        walk_directory(self.root_path)
        return sorted(filtered_paths, key=lambda p: (p.is_file(), str(p).lower()))

    def generate_tree(self, filtered_paths: List[Path]) -> str:
        """ 使用过滤后的路径列表生成目录树。 """
        tree_lines = [f"{self.root_path.name}/"]
        
        # 创建一个 path -> list of children 的映射
        tree_map = {self.root_path: []}
        for path in filtered_paths:
            parent = path.parent
            if parent not in tree_map:
                tree_map[parent] = []
            tree_map[parent].append(path)

        def build_recursive(directory: Path, prefix: str):
            if directory not in tree_map:
                return
            
            children = tree_map[directory]
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{child.name}")
                if child.is_dir():
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    build_recursive(child, new_prefix)
        
        build_recursive(self.root_path, "")
        return "```\n" + "\n".join(tree_lines) + "\n```"

    def _escape_markdown_content(self, content: str) -> str:
        """转义markdown内容中的特殊字符，防止渲染冲突"""
        # 转义代码块标记
        content = content.replace("```", "\\`\\`\\`")
        # 转义其他可能冲突的markdown语法
        content = content.replace("# ", "\\# ")
        content = content.replace("## ", "\\## ")
        content = content.replace("### ", "\\### ")
        return content

    def generate_content_summary(self, filtered_paths: List[Path]) -> str:
        """ 使用过滤后的路径列表生成文件内容。 """
        content_str = ""
        file_paths = [p for p in filtered_paths if p.is_file()]

        for file_path in file_paths:
            relative_path_str = file_path.relative_to(self.root_path).as_posix()
            print(f"正在处理文件: {relative_path_str}")

            # 获取文件扩展名
            file_extension = file_path.suffix.lower()
            
            # 确定语言标识
            lang = LANGUAGE_MAP.get(file_extension, "")
            if file_path.name in LANGUAGE_MAP:
                lang = LANGUAGE_MAP[file_path.name]
            
            content_str += f"## `{relative_path_str}`\n\n"
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if not content:
                    content_str += "[文件为空]\n\n"
                    continue
                
                # 对特殊文件类型进行处理
                if file_extension in SPECIAL_HANDLING_EXTENSIONS:
                    # 对于markdown等文本文件，使用转义处理
                    escaped_content = self._escape_markdown_content(content)
                    content_str += f"````{lang}\n{escaped_content}\n````\n\n"
                else:
                    # 对于代码文件，正常处理
                    content_str += f"```{lang}\n{content}\n```\n\n"
                    
            except Exception as e:
                content_str += f"[无法读取文件内容: {str(e)}]\n\n"
                
        return content_str

    def generate_full_overview(self, output_file: str):
        print("开始生成项目概览...")
        
        # 1. 首先，一次性获取所有需要处理的路径
        print("正在过滤项目路径...")
        filtered_paths = self.get_filtered_paths()
        
        # 2. 基于过滤后的路径生成目录树
        print("正在生成目录树...")
        tree = self.generate_tree(filtered_paths)

        # 3. 基于过滤后的路径生成文件内容
        print("正在聚合文件内容...")
        contents = self.generate_content_summary(filtered_paths)
        
        final_markdown = (
            f"# 项目概览: {self.root_path.name}\n\n"
            f"本文档由`{SCRIPT_NAME}`自动生成，包含了项目的结构树和所有可读文件的内容。\n\n"
            f"## 项目结构\n\n{tree}\n\n---\n\n"
            f"# 文件内容\n\n{contents}"
        )
        
        try:
            with open(self.root_path / output_file, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            print(f"\n🎉 项目概览生成成功！文件已保存至: {self.root_path / output_file}")
        except IOError as e:
            print(f"\n❌ 错误：无法写入文件 {output_file}。详情: {e}")

def main():
    project_root = os.getcwd() 
    documenter = ProjectDocumenter(
        root_dir=project_root,
        ignore_patterns=IGNORE_PATTERNS
    )
    documenter.generate_full_overview(OUTPUT_FILENAME)

if __name__ == "__main__":
    main()