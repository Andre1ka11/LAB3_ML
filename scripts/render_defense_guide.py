"""Optional Windows PDF export using an installed Edge and Playwright."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]


def render():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1300,'height':1500})
        page.goto((ROOT/'docs/defense_guide.html').as_uri(),wait_until='load');page.evaluate('document.fonts.ready')
        page.pdf(path=str(ROOT/'docs/defense_guide.pdf'),format='A4',print_background=True,prefer_css_page_size=True,
            display_header_footer=True,header_template='<span></span>',
            footer_template='<div style="font-size:9px;color:#687382;text-align:center;width:100%"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    return ROOT/'docs/defense_guide.pdf'


if __name__=='__main__':print(render())
