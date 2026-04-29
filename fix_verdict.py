import re

with open('frontend/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the verdict reference
old = '{result?.verdict ?? "A clean, premium summary of the product\'s trust signals."}'
new = '{verdict}'
content = content.replace(old, new)

with open('frontend/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed verdict reference")
