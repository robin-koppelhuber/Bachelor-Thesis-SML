"""Registry for merging methods"""

from typing import Callable, Dict, List, Type

from src.methods.base import BaseMergingMethod


class MethodRegistry:
    """Registry for merging methods"""

    _methods: Dict[str, Type[BaseMergingMethod]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a method

        Args:
            name: Method name

        Returns:
            Decorator function

        Example:
            @MethodRegistry.register("ties")
            class TIESMerging(BaseMergingMethod):
                ...
        """

        def decorator(method_class: Type[BaseMergingMethod]):
            cls._methods[name] = method_class
            return method_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseMergingMethod]:
        """
        Get method class by name

        Args:
            name: Method name

        Returns:
            Method class

        Raises:
            ValueError: If method not found
        """
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}. Available: {list(cls._methods.keys())}")
        return cls._methods[name]

    @classmethod
    def list_methods(cls) -> List[str]:
        """
        List all registered methods

        Returns:
            List of method names
        """
        return list(cls._methods.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseMergingMethod:
        """
        Create method instance

        Args:
            name: Method name
            **kwargs: Method parameters

        Returns:
            Method instance
        """
        method_class = cls.get(name)
        return method_class(**kwargs)
