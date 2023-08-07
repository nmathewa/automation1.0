from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

url = 'https://www.wunderground.com/history/daily/us/fl/melbourne/KMLB/date/2023-08-07'
driver = webdriver.Chrome()
driver.get(url)
tables = WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
for table in tables:
    newTable = pd.read_html(table.get_attribute('outerHTML'))
    if newTable:
        print(newTable[0].fillna(''))