# from tinytroupe.agent import TinyPerson
# from tinytroupe.examples import (
#     create_lisa_the_data_scientist,
#     create_oscar_the_architect,
# )
#
# import os
#
# os.environ["OPENAI_API_KEY"] = (
#     "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"
# )
#
# # User search query: "55 inches tv"
#
# tv_ad_1 = """
# The Best TV Of Tomorrow - LG 4K Ultra HD TV
# https://www.lg.com/tv/oled
# AdThe Leading Name in Cinematic Picture. Upgrade Your TV to 4K OLED And See The Difference. It's Not Just OLED, It's LG OLED. Exclusive a9 Processor, Bringing Cinematic Picture Home.
#
# Infinite Contrast · Self-Lighting OLED · Dolby Vision™ IQ · ThinQ AI w/ Magic Remote
#
# Free Wall Mounting Deal
# LG G2 97" OLED evo TV
# Free TV Stand w/ Purchase
# World's No.1 OLED TV
# """
#
# tv_ad_2 = """
# The Full Samsung TV Lineup - Neo QLED, OLED, 4K, 8K & More
# https://www.samsung.com
# AdFrom 4K To 8K, QLED To OLED, Lifestyle TVs & More, Your Perfect TV Is In Our Lineup. Experience Unrivaled Technology & Design In Our Ultra-Premium 8K & 4K TVs.
#
# Discover Samsung Event · Real Depth Enhancer · Anti-Reflection · 48 mo 0% APR Financing
#
# The 2023 OLED TV Is Here
# Samsung Neo QLED 4K TVs
# Samsung Financing
# Ranked #1 By The ACSI®
# """
#
# tv_ad_3 = """
# Wayfair 55 Inch Tv - Wayfair 55 Inch Tv Décor
# Shop Now
# https://www.wayfair.com/furniture/free-shipping
# AdFree Shipping on Orders Over $35. Shop Furniture, Home Décor, Cookware & More! Free Shipping on All Orders Over $35. Shop 55 Inch Tv, Home Décor, Cookware & More!
# """
#
# eval_request_msg = f"""
# Can you evaluate these Bing ads for me? Which one convices you more to buy their particular offering?
# Select **ONLY** one. Please explain your reasoning, based on your financial situation, background and personality.
#
# # AD 1
# ```
# {tv_ad_1}
# ```
#
# # AD 2
# ```
# {tv_ad_2}
# ```
#
# # AD 3
# ```
# {tv_ad_3}
# ```
# """
#
# print(eval_request_msg)
#
# situation = "Your TV broke and you need a new one. You search for a new TV on Bing."
#
# lisa = create_lisa_the_data_scientist()
#
# lisa.change_context(situation)
#
# lisa.listen_and_act(eval_request_msg)


from marqetsim import cli
cli.launch()
# import sys
# sys.argv = ['marq', 'agents/situations/test-situation-1.yaml']
