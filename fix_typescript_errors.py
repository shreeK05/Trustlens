#!/usr/bin/env python3
"""Fix remaining TypeScript errors in frontend/src/App.tsx"""

import re

# Read the file
with open('frontend/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Hero section verdict reference (line 712)
content = content.replace(
    '{result?.verdict ?? "A clean, premium summary of the product\'s trust signals."}',
    '{verdict}'
)

# Fix 2: Source mode reference (line 714)
content = content.replace(
    '{result?.source_mode === "live_scrape"',
    '{result?.product'
)

# Fix 3: Price anomaly trend (line 994)
old_trend = '{result.price_anomaly.price_trend.replace(/_/g, " ")}'
new_trend = '{safeGet(result, "ml_results.price_anomaly.price_trend", "stable").replace(/_/g, " ")}'
content = content.replace(old_trend, new_trend)

# Fix 4: Price anomaly discount (line 1002)
old_discount = '{formatPercent(result.price_anomaly.discount_pct)}'
new_discount = '{formatPercent(safeGet(result, "ml_results.price_anomaly.discount_pct", 0))}'
content = content.replace(old_discount, new_discount)

# Fix 5: Current vs MRP (line 998) - multiple occurrences
old_mrp_price = '{result.mrp > result.price ? `${formatPercent(((result.mrp - result.price) / result.mrp) * 100)} off` : "No markdown"}'
new_mrp_price = '{productMrp > productPrice ? `${formatPercent(((productMrp - productPrice) / productMrp) * 100)} off` : "No markdown"}'
content = content.replace(old_mrp_price, new_mrp_price)

# Fix 6: Competitor prices - replace result.competitor_prices with safeGet
content = re.sub(
    r'\{result\.competitor_prices && result\.competitor_prices\.length > 0 &&',
    '{safeGet(result, "competitor_prices", []).length > 0 &&',
    content
)

# Fix 7: Competitor prices length
content = content.replace(
    'Compared {result.competitor_prices.length + 1} trusted',
    'Compared {safeGet(result, "competitor_prices", []).length + 1} trusted'
)

# Fix 8: Competitor price map
content = content.replace(
    '{result.competitor_prices.map(comp =>',
    '{safeGet(result, "competitor_prices", []).map((comp: {platform: string; price: number}) =>'
)

# Fix 9: Current price in competitor section
content = re.sub(
    r'\{formatRupee\(result\.price\)\}(.*?)Your current price',
    '{formatRupee(productPrice)}\g<1>Your current price',
    content,
    flags=re.DOTALL
)

# Fix 10: Competitor price comparisons (multiple locations)
content = content.replace('comp.price < result.price', 'comp.price < productPrice')
content = content.replace('comp.price === result.price', 'comp.price === productPrice')

# Fix 11: Save/more calculations
content = content.replace('(result.price - comp.price)', '(productPrice - comp.price)')
content = content.replace('(comp.price - result.price)', '(comp.price - productPrice)')

print("Fixes applied successfully!")

# Write the file back
with open('frontend/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("File updated: frontend/src/App.tsx")
