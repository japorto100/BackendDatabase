import os
import tomli
from django.conf import settings

class ConfigHandler:
    """
    Handler for reading and parsing TOML configuration files
    """
    def __init__(self, config_path=None):
        if config_path is None:
            self.config_path = os.path.join(settings.BASE_DIR, 'config.toml')
        else:
            self.config_path = config_path
        
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from TOML file"""
        try:
            with open(self.config_path, 'rb') as f:
                return tomli.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {self.config_path}")
            return {}
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return {}
    
    def get(self, section, key, default=None):
        """
        Get a configuration value
        
        Args:
            section: Section name in the TOML file
            key: Key name within the section
            default: Default value if the key is not found
            
        Returns:
            The configuration value or the default
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def get_section(self, section, default=None):
        """
        Get an entire configuration section
        
        Args:
            section: Section name in the TOML file
            default: Default value if the section is not found
            
        Returns:
            The section as a dictionary or the default
        """
        if default is None:
            default = {}
        
        try:
            return self.config.get(section, default)
        except Exception:
            return default

# Create a singleton instance
config = ConfigHandler() 