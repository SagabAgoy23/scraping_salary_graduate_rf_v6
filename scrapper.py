import time
from dataclasses import dataclass, asdict
from typing import List
from urllib.parse import quote_plus, urljoin

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

import config


@dataclass
class JobRecord:
    source_keyword: str
    job_title: str
    company: str
    location: str
    salary_text: str
    job_description: str
    job_url: str
    scraped_at: str


class JobStreetScraper:
    def __init__(self, headless: bool = config.HEADLESS):
        self.headless = headless
        self.driver = None
        self.restart_count = 0

    def _build_driver(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1366,768")
        options.add_argument("--lang=id-ID")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--mute-audio")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(config.SELENIUM_PAGE_LOAD_TIMEOUT)
        return driver

    def __enter__(self):
        self.driver = self._build_driver()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._quit_driver()

    def _quit_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def _restart_driver(self):
        self.restart_count += 1
        if self.restart_count > config.MAX_DRIVER_RESTARTS:
            raise RuntimeError(
                f"ChromeDriver sudah restart {self.restart_count} kali. "
                "Scraping dihentikan agar tidak loop tanpa akhir."
            )
        print(f"ChromeDriver mati/tidak valid. Restart browser... ({self.restart_count}/{config.MAX_DRIVER_RESTARTS})")
        self._quit_driver()
        time.sleep(config.RESTART_DELAY_SECONDS)
        self.driver = self._build_driver()

    def _ensure_driver(self):
        if self.driver is None:
            self.driver = self._build_driver()

    def _search_url(self, keyword: str, page: int) -> str:
        encoded = quote_plus(keyword)
        return f"{config.JOBSTREET_BASE_URL}/jobs?keywords={encoded}&page={page}"

    def _load_page(self, url: str, delay: int) -> bool:
        """Load halaman dengan retry. Return False jika halaman tetap gagal dibuka."""
        for attempt in range(1, config.MAX_PAGE_RETRIES + 1):
            try:
                self._ensure_driver()
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 20).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                except TimeoutException:
                    pass
                time.sleep(delay)
                return True
            except TimeoutException:
                print(f"Timeout saat membuka halaman, lanjut pakai HTML yang sempat termuat. Attempt {attempt}.")
                try:
                    self.driver.execute_script("window.stop();")
                except Exception:
                    pass
                time.sleep(delay)
                return True
            except (InvalidSessionIdException, WebDriverException) as exc:
                print(f"Gagal membuka halaman. Attempt {attempt}/{config.MAX_PAGE_RETRIES}. Error: {type(exc).__name__}")
                self._restart_driver()
                time.sleep(config.RESTART_DELAY_SECONDS)
        print(f"Halaman dilewati karena gagal dibuka: {url}")
        return False

    def _scroll_page(self) -> bool:
        try:
            last_height = 0
            for _ in range(6):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(config.SCROLL_PAUSE_SECONDS)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            return True
        except (InvalidSessionIdException, WebDriverException) as exc:
            print(f"Scroll gagal karena session browser rusak: {type(exc).__name__}")
            self._restart_driver()
            return False

    def _get_page_source(self) -> str:
        try:
            return self.driver.page_source
        except (InvalidSessionIdException, WebDriverException) as exc:
            print(f"Gagal membaca page_source: {type(exc).__name__}")
            self._restart_driver()
            return ""

    @staticmethod
    def _clean_text(value: str) -> str:
        if not value:
            return ""
        return " ".join(str(value).split())

    @staticmethod
    def _pick_text(card, selectors: list[str]) -> str:
        if not card:
            return ""
        for selector in selectors:
            el = card.select_one(selector)
            if el and el.get_text(strip=True):
                return JobStreetScraper._clean_text(el.get_text(" ", strip=True))
        return ""

    @staticmethod
    def _extract_salary_from_text(text: str) -> str:
        """Fallback ringan jika selector salary JobStreet berubah."""
        if not text:
            return ""
        lowered = text.lower()
        salary_markers = ["rp", "idr", "juta", "jt", "per bulan", "per month"]
        if not any(marker in lowered for marker in salary_markers):
            return ""
        tokens = text.split()
        chunks = []
        for i, token in enumerate(tokens):
            t = token.lower()
            if any(marker in t for marker in ["rp", "idr", "juta", "jt"]):
                start = max(0, i - 6)
                end = min(len(tokens), i + 10)
                chunks.append(" ".join(tokens[start:end]))
        return " | ".join(chunks[:3])

    def _parse_cards(self, html: str, keyword: str) -> List[dict]:
        soup = BeautifulSoup(html, "lxml")
        cards = []

        title_links = []
        for anchor in soup.select('a[href*="/job/"]'):
            title = anchor.get_text(" ", strip=True)
            href = anchor.get("href")
            if title and href and len(title) >= 3:
                title_links.append(anchor)

        seen_urls = set()
        for anchor in title_links:
            url = urljoin(config.JOBSTREET_BASE_URL, anchor.get("href"))
            if url in seen_urls:
                continue
            seen_urls.add(url)

            card = anchor.find_parent("article") or anchor.find_parent("div") or anchor
            for _ in range(8):
                if card and len(card.get_text(" ", strip=True)) < 180:
                    card = card.find_parent("div")
                else:
                    break

            card_text = self._clean_text(card.get_text(" ", strip=True)) if card else ""
            title = self._clean_text(anchor.get_text(" ", strip=True))
            company = self._pick_text(card, [
                '[data-automation="jobCompany"]',
                '[data-testid="company-name"]',
                'span[data-automation*="company"]',
                'a[data-automation*="company"]',
            ])
            location = self._pick_text(card, [
                '[data-automation="jobLocation"]',
                '[data-testid="job-location"]',
                'span[data-automation*="location"]',
            ])
            salary = self._pick_text(card, [
                '[data-automation="jobSalary"]',
                '[data-testid="job-salary"]',
                'span[data-automation*="salary"]',
                '[aria-label*="salary"]',
                '[aria-label*="gaji"]',
            ])
            if not salary:
                salary = self._extract_salary_from_text(card_text)

            cards.append({
                "source_keyword": keyword,
                "job_title": title,
                "company": company,
                "location": location,
                "salary_text": salary,
                "card_text": card_text,
                "job_url": url,
            })

        return cards

    def _fetch_detail_text(self, url: str) -> str:
        if not config.FETCH_JOB_DETAIL:
            return ""
        try:
            loaded = self._load_page(url, config.DETAIL_LOAD_DELAY_SECONDS)
            if not loaded:
                return ""
            detail_html = self._get_page_source()
            if not detail_html:
                return ""
            soup = BeautifulSoup(detail_html, "lxml")

            selectors = [
                '[data-automation="jobAdDetails"]',
                '[data-automation="jobDescription"]',
                '[data-testid="job-description"]',
                'section[data-automation*="job"]',
                'main',
            ]
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    text = self._clean_text(element.get_text(" ", strip=True))
                    if len(text) > 100:
                        return text[:12000]
            return self._clean_text(soup.get_text(" ", strip=True))[:12000]
        except (InvalidSessionIdException, WebDriverException) as exc:
            print(f"Detail lowongan gagal dibaca, dilewati: {type(exc).__name__}")
            self._restart_driver()
            return ""
        except Exception as exc:
            print(f"Detail lowongan gagal dibaca, dilewati: {type(exc).__name__}")
            return ""

    def _load_seen_urls(self) -> set:
        seen_urls = set()
        paths = []
        if config.SKIP_EXISTING_URLS and config.RAW_OUTPUT_FILE.exists():
            paths.append(config.RAW_OUTPUT_FILE)
        if config.SESSION_CHECKPOINT_FILE.exists():
            paths.append(config.SESSION_CHECKPOINT_FILE)

        for path in paths:
            try:
                df = pd.read_csv(path)
                if "job_url" in df.columns:
                    seen_urls.update(df["job_url"].dropna().astype(str).tolist())
            except Exception:
                pass
        return seen_urls

    def _save_checkpoint(self, records: List[dict]):
        if not records:
            return
        try:
            config.OUTPUT_DIR.mkdir(exist_ok=True)
            df = pd.DataFrame(records)
            if "job_url" in df.columns:
                df = df.drop_duplicates(subset=["job_url"], keep="last")
            df.to_csv(config.SESSION_CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
            print(f"Checkpoint tersimpan: {config.SESSION_CHECKPOINT_FILE} ({len(df)} baris sesi ini)")
        except Exception as exc:
            print(f"Gagal menyimpan checkpoint: {type(exc).__name__}: {exc}")

    def scrape(self) -> List[dict]:
        all_records = []
        seen_urls = self._load_seen_urls()
        print(f"URL lama/checkpoint yang akan dilewati: {len(seen_urls)}")

        for keyword_index, keyword in enumerate(config.SEARCH_KEYWORDS, start=1):
            print(f"\n[{keyword_index}/{len(config.SEARCH_KEYWORDS)}] Scraping keyword: {keyword}")
            collected_for_keyword = 0
            empty_pages = 0

            for page in range(1, config.MAX_PAGE_PER_KEYWORD + 1):
                if collected_for_keyword >= config.MAX_JOBS_PER_KEYWORD:
                    break

                url = self._search_url(keyword, page)
                print(f"Open page {page}: {url}")
                loaded = self._load_page(url, config.PAGE_LOAD_DELAY_SECONDS)
                if not loaded:
                    empty_pages += 1
                    if empty_pages >= 2:
                        print("Dua halaman gagal/kosong berturut-turut. Pindah keyword berikutnya.")
                        break
                    continue

                self._scroll_page()
                html = self._get_page_source()
                if not html:
                    empty_pages += 1
                    continue

                cards = self._parse_cards(html, keyword)
                if not cards:
                    empty_pages += 1
                    print("Tidak ada kartu lowongan terbaca pada halaman ini.")
                    if empty_pages >= 2:
                        print("Dua halaman kosong berturut-turut. Pindah keyword berikutnya.")
                        break
                    continue

                before_count = collected_for_keyword
                for card in cards:
                    if collected_for_keyword >= config.MAX_JOBS_PER_KEYWORD:
                        break
                    if card["job_url"] in seen_urls:
                        continue

                    seen_urls.add(card["job_url"])
                    detail_text = self._fetch_detail_text(card["job_url"])
                    description = self._clean_text(f"{card.get('card_text', '')} {detail_text}")

                    record = JobRecord(
                        source_keyword=card.get("source_keyword", ""),
                        job_title=card.get("job_title", ""),
                        company=card.get("company", ""),
                        location=card.get("location", ""),
                        salary_text=card.get("salary_text", ""),
                        job_description=description,
                        job_url=card.get("job_url", ""),
                        scraped_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    all_records.append(asdict(record))
                    collected_for_keyword += 1
                    print(f"Collected {collected_for_keyword}: {record.job_title[:90]}")

                    if len(all_records) % config.CHECKPOINT_EVERY_N_RECORDS == 0:
                        self._save_checkpoint(all_records)

                self._save_checkpoint(all_records)

                if collected_for_keyword == before_count:
                    empty_pages += 1
                    print("Halaman terbaca, tetapi semua URL duplikat atau tidak bisa dipakai.")
                    if empty_pages >= 3:
                        print("Tiga halaman tanpa data baru. Pindah keyword berikutnya.")
                        break
                else:
                    empty_pages = 0

        self._save_checkpoint(all_records)
        return all_records
