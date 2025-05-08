"""
Utility functions for text processing in the AI VTuber application.
"""

def remove_special_characters(text):
    """
    Remove special characters from text to prevent the AI from reading them.

    Args:
        text (str): The input text that may contain special characters

    Returns:
        str: The text with special characters removed
    """
    # Define a list of special characters to remove
    # This includes various symbols, emojis, and other special characters
    special_chars = [
        '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
        '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\',
        ':', ';', '"', "'", '<', '>', ',', '.', '?', '/',
        '~', '`', '•', '★', '☆', '♥', '♦', '♣', '♠', '♪',
        '♫', '☼', '►', '◄', '▲', '▼', '■', '□', '▪', '▫',
        '❤', '❥', '❦', '❧', '☜', '☞', '☝', '☟', '✌', '✍',
        '✎', '✏', '✐', '✑', '✒', '✓', '✔', '✕', '✖', '✗',
        '✘', '✙', '✚', '✛', '✜', '✝', '✞', '✟', '✠', '✡',
        '✢', '✣', '✤', '✥', '✦', '✧', '✩', '✪', '✫', '✬',
        '✭', '✮', '✯', '✰', '✱', '✲', '✳', '✴', '✵', '✶',
        '✷', '✸', '✹', '✺', '✻', '✼', '✽', '✾', '✿', '❀',
        '❁', '❂', '❃', '❄', '❅', '❆', '❇', '❈', '❉', '❊',
        '❋', '❌', '❍', '❎', '❏', '❐', '❑', '❒', '▬', '▭',
        '▮', '▯', '▰', '▱', '▶', '◀', '◢', '◣', '◤', '◥',
        '☀', '☁', '☂', '☃', '☄', '★', '☆', '☇', '☈', '☉',
        '☊', '☋', '☌', '☍', '☎', '☏', '☐', '☑', '☒', '☓',
        '☔', '☕', '☖', '☗', '☘', '☙', '☚', '☛', '☜', '☝',
        '☞', '☟', '☠', '☡', '☢', '☣', '☤', '☥', '☦', '☧',
        '☨', '☩', '☪', '☫', '☬', '☭', '☮', '☯', '☰', '☱',
        '☲', '☳', '☴', '☵', '☶', '☷', '☸', '☹', '☺', '☻',
        '☼', '☽', '☾', '☿', '♀', '♁', '♂', '♃', '♄', '♅',
        '♆', '♇', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏',
        '♐', '♑', '♒', '♓', '♔', '♕', '♖', '♗', '♘', '♙',
        '♚', '♛', '♜', '♝', '♞', '♟', '♠', '♡', '♢', '♣',
        '♤', '♥', '♦', '♧', '♨', '♩', '♪', '♫', '♬', '♭',
        '♮', '♯', '♰', '♱', '♲', '♳', '♴', '♵', '♶', '♷',
        '♸', '♹', '♺', '♻', '♼', '♽', '♾', '♿'
    ]

    # Create a translation table to remove special characters
    translation_table = str.maketrans('', '', ''.join(special_chars))

    # Apply the translation table to remove special characters
    filtered_text = text.translate(translation_table)

    return filtered_text


