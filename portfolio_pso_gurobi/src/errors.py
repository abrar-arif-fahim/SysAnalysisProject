class PortfolioError(Exception):
    """Base exception for portfolio project."""
    pass


class ConfigError(PortfolioError):
    """Raised when configuration loading/parsing fails."""
    pass


class DataError(PortfolioError):
    """Raised when data fetching or generation fails."""
    pass


class GurobiError(PortfolioError):
    """Raised when Gurobi import or optimization fails."""
    pass


class PSOError(PortfolioError):
    """Raised for PSO-specific failures."""
    pass
