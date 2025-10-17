import os
import sys
import time
import random
import requests
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver


MAX_MB = int(sys.argv[1]) if len(sys.argv) > 1 else None
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 123
random.seed(SEED)

MAX_BYTES = MAX_MB * 1024 * 1024 if MAX_MB else float('inf')
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Limite definido: {MAX_MB if MAX_MB else 'sem limite'} MB\n")

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

def download_file(url, filename, total_downloaded):
    output_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Baixando o livro de código '{filename}'")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue

                if MAX_MB is not None and total_downloaded + len(chunk) > MAX_BYTES:
                    remaining = MAX_BYTES - total_downloaded
                    if remaining > 0:
                        f.write(chunk[:remaining])
                        progress_bar.update(remaining)
                        total_downloaded += remaining
                    progress_bar.close()
                    print(f"\nLimite de {MAX_MB} MB atingido durante {filename}. Interrompendo downloads.")
                    return total_downloaded, True, False

                f.write(chunk)
                downloaded += len(chunk)
                total_downloaded += len(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()
        return total_downloaded, False, True

    except requests.exceptions.RequestException as e:
        print(f"ERRO ao baixar {filename}: {e}")
        return total_downloaded, False, False


def get_pt_books():
    print("Buscando livros em português...")
    url = "https://www.gutenberg.org/browse/languages/pt"
    driver.get(url)
    time.sleep(2)

    div = driver.find_element(By.CLASS_NAME, "pgdbbylanguage")
    uls = div.find_elements(By.TAG_NAME, "ul")

    base_links = []
    for ul in uls:
        lis = ul.find_elements(By.TAG_NAME, "li")
        for li in lis:
            if "English" not in li.text:
                try:
                    a = li.find_element(By.TAG_NAME, "a")
                    href = a.get_attribute("href")
                    if href and href.startswith("https://www.gutenberg.org/ebooks/"):
                        base_links.append(href)
                except Exception:
                    continue

    print(f"{len(base_links)} links base coletados.\n")
    random.shuffle(base_links)

    total_downloaded = 0
    successful = 0

    for link in base_links:
        codigo = link.split("/")[-1]
        url_livro = f"https://www.gutenberg.org/files/{codigo}"
        try:
            driver.get(url_livro)
            anchors = driver.find_elements(By.TAG_NAME, "a")
            for a in anchors:
                href = a.get_attribute("href")
                if href and href.endswith(".txt"):
                    print(f"Livro encontrado: {href}")
                    total_downloaded, stop, success = download_file(href, codigo, total_downloaded)
                    if success:
                        successful += 1
                    if stop:
                        print("Encerrando processo por atingir o limite definido.")
                        print(f"Total de livros baixados com sucesso: {successful}")
                        print(f"Tamanho total baixado: {total_downloaded / 1024 / 1024:.2f} MB")
                        return
                    break
        except Exception as e:
            print(f"Erro em {url_livro}: {e}")

    print(f"Todos os livros disponíveis foram baixados com sucesso: {successful}")
    print(f"Tamanho total geral baixado: {total_downloaded / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    try:
        get_pt_books()
    finally:
        driver.quit()