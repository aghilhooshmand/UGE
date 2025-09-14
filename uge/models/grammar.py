"""
Grammar Model for UGE Application

This module defines the Grammar data model, which represents BNF grammars
used in Grammatical Evolution experiments.

Classes:
- Grammar: Main grammar data model
- GrammarInfo: Grammar metadata and information

Author: UGE Team
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GrammarInfo:
    """
    Grammar metadata and information.
    
    This class contains metadata about a BNF grammar file, including its name,
    path, size, and basic information.
    
    Attributes:
        name (str): Grammar name
        path (Path): Path to the grammar file
        size_bytes (int): File size in bytes
        lines (int): Number of lines in the grammar
        rules (int): Number of grammar rules
        terminals (List[str]): List of terminal symbols
        non_terminals (List[str]): List of non-terminal symbols
    """
    
    name: str
    path: Path
    size_bytes: int
    lines: int
    rules: int
    terminals: List[str]
    non_terminals: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert grammar info to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'size_bytes': self.size_bytes,
            'lines': self.lines,
            'rules': self.rules,
            'terminals': self.terminals,
            'non_terminals': self.non_terminals
        }


class Grammar:
    """
    Grammar data model for UGE experiments.
    
    This class represents a BNF grammar used in Grammatical Evolution experiments.
    It handles loading, parsing, and validation of grammar files.
    
    Attributes:
        info (GrammarInfo): Grammar metadata
        content (Optional[str]): Grammar file content
        rules (Dict[str, List[str]]): Parsed grammar rules
    """
    
    def __init__(self, info: GrammarInfo):
        """
        Initialize grammar with metadata.
        
        Args:
            info (GrammarInfo): Grammar metadata
        """
        self.info = info
        self.content: Optional[str] = None
        self.rules: Dict[str, List[str]] = {}
    
    def load(self) -> str:
        """
        Load grammar content from file.
        
        Returns:
            str: Grammar file content
            
        Raises:
            FileNotFoundError: If grammar file doesn't exist
            ValueError: If grammar file cannot be read
        """
        if not self.info.path.exists():
            raise FileNotFoundError(f"Grammar file not found: {self.info.path}")
        
        try:
            with open(self.info.path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            return self.content
        except Exception as e:
            raise ValueError(f"Error reading grammar file: {e}")
    
    def parse(self) -> Dict[str, List[str]]:
        """
        Parse grammar content into rules.
        
        This method parses the BNF grammar content and extracts the rules.
        It handles the standard BNF format with ::= as the rule separator.
        
        Returns:
            Dict[str, List[str]]: Parsed grammar rules
            
        Raises:
            ValueError: If grammar format is invalid
        """
        if self.content is None:
            self.load()
        
        rules = {}
        lines = self.content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            if '::=' in line:
                parts = line.split('::=', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid grammar rule format: {line}")
                
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                # Handle multiple alternatives separated by |
                alternatives = [alt.strip() for alt in rhs.split('|')]
                rules[lhs] = alternatives
        
        self.rules = rules
        return rules
    
    def validate(self) -> List[str]:
        """
        Validate grammar for common issues.
        
        Returns:
            List[str]: List of validation warnings/errors
        """
        warnings = []
        
        if not self.rules:
            self.parse()
        
        # Check for empty rules
        for lhs, alternatives in self.rules.items():
            if not alternatives:
                warnings.append(f"Empty rule for {lhs}")
            
            for alt in alternatives:
                if not alt.strip():
                    warnings.append(f"Empty alternative in rule {lhs}")
        
        # Check for undefined non-terminals
        defined_symbols = set(self.rules.keys())
        used_symbols = set()
        
        for alternatives in self.rules.values():
            for alt in alternatives:
                # Simple extraction of symbols (this could be more sophisticated)
                symbols = [s.strip() for s in alt.split() if s.strip()]
                for symbol in symbols:
                    if symbol.startswith('<') and symbol.endswith('>'):
                        used_symbols.add(symbol)
        
        undefined = used_symbols - defined_symbols
        if undefined:
            warnings.append(f"Undefined non-terminals: {', '.join(undefined)}")
        
        return warnings
    
    def get_terminals(self) -> List[str]:
        """
        Extract terminal symbols from grammar.
        
        Returns:
            List[str]: List of terminal symbols
        """
        if not self.rules:
            self.parse()
        
        terminals = set()
        for alternatives in self.rules.values():
            for alt in alternatives:
                symbols = [s.strip() for s in alt.split() if s.strip()]
                for symbol in symbols:
                    if not (symbol.startswith('<') and symbol.endswith('>')):
                        terminals.add(symbol)
        
        return sorted(list(terminals))
    
    def get_non_terminals(self) -> List[str]:
        """
        Extract non-terminal symbols from grammar.
        
        Returns:
            List[str]: List of non-terminal symbols
        """
        if not self.rules:
            self.parse()
        
        return sorted(list(self.rules.keys()))
    
    def get_start_symbol(self) -> Optional[str]:
        """
        Get the start symbol of the grammar.
        
        Returns:
            Optional[str]: Start symbol if found
        """
        if not self.rules:
            self.parse()
        
        # Common start symbols
        start_candidates = ['<start>', '<S>', '<expr>', '<expression>']
        
        for candidate in start_candidates:
            if candidate in self.rules:
                return candidate
        
        # If no common start symbol found, return the first rule
        if self.rules:
            return list(self.rules.keys())[0]
        
        return None
    
    def get_preview(self, n_lines: int = 20) -> str:
        """
        Get a preview of the grammar content.
        
        Args:
            n_lines (int): Number of lines to preview
            
        Returns:
            str: Grammar preview
        """
        if self.content is None:
            self.load()
        
        lines = self.content.split('\n')
        return '\n'.join(lines[:n_lines])
    
    def save(self, content: str) -> None:
        """
        Save grammar content to file.
        
        Args:
            content (str): Grammar content to save
            
        Raises:
            ValueError: If content cannot be saved
        """
        try:
            with open(self.info.path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.content = content
            # Update info
            self.info.size_bytes = len(content.encode('utf-8'))
            self.info.lines = len(content.split('\n'))
        except Exception as e:
            raise ValueError(f"Error saving grammar file: {e}")
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Grammar':
        """
        Create grammar from file path.
        
        Args:
            file_path (Path): Path to grammar file
            
        Returns:
            Grammar: Grammar instance
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        size_bytes = file_path.stat().st_size
        
        # Count lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        
        # Create info (terminals and non-terminals will be populated after parsing)
        info = GrammarInfo(
            name=file_path.name,
            path=file_path,
            size_bytes=size_bytes,
            lines=lines,
            rules=0,  # Will be updated after parsing
            terminals=[],
            non_terminals=[]
        )
        
        grammar = cls(info)
        
        # Parse to get additional info
        try:
            grammar.parse()
            info.rules = len(grammar.rules)
            info.terminals = grammar.get_terminals()
            info.non_terminals = grammar.get_non_terminals()
        except Exception as e:
            # If parsing fails, we still have basic info
            pass
        
        return grammar