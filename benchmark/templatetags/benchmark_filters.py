from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using a key"""
    if not dictionary:
        return None
    return dictionary.get(str(key)) if isinstance(key, int) else dictionary.get(key)

@register.filter
def get_color(strategy_name, alpha=1.0):
    """Get a color for a strategy name"""
    colors = {
        'early': f'rgba(54, 162, 235, {alpha})',
        'late': f'rgba(255, 99, 132, {alpha})',
        'attention': f'rgba(255, 206, 86, {alpha})',
        'hybrid': f'rgba(75, 192, 192, {alpha})'
    }
    return colors.get(strategy_name.lower(), f'rgba(201, 203, 207, {alpha})') 