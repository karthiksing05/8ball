# Web Scraping Imports
from selenium import webdriver

# Reg Python Imports
import pickle

URL = "https://robinhood.com/us/en/"
CREDS = "credentials.pickle"


def _get_username_password():
    username = input("Username: ")
    password = input("Password: ")

    with open(CREDS, "wb") as f:
        pickle.dump([username, password, f])

def _log_in():
    """
    A helper function to log in, for cleaner code. 
    """

    with open(CREDS, "") as f:
        creds_list = pickle.load(f)
        username = creds_list[0]
        password = creds_list[1]

    driver = webdriver.Chrome(executable_path="C:\\Program Files (x86)\\chromedriver.exe")
    driver.get(URL)

    login_box = driver.find_element_by_class_name("css-13gbyvh").find_element_by_class_name("css-1pnpn8p").find_element_by_class_name("css-1j7oy5d-UnstyledAnchor-Button")
    login_box.click()

    username_box = driver.find_element_by_class_name("css-1ohcxfz").find_element_by_class_name("css-8atqhb").find_element_by_class_name("css-yvh4hz-InternalInput").find_element_by_class_name("remove-legacy css-9psyor-InternalInput")
    username_box.send_keys(username)

    password_box = driver.find_element_by_class_name("css-17nqzf1").find_element_by_class_name("css-8atqhb").find_element_by_class_name("css-8l1bak-InternalInput").find_element_by_class_name("remove-legacy css-j4zbll-InternalInput")
    password_box.send_keys(password)

    enter_box = driver.find_element_by_class_name("css-tp596t").find_element_by_class_name("css-0").find_element_by_class_name("_1OsoaRGpMCXh9KT8s7wtwm rhLoginButton _2GHn41jUsfSSC9HmVWT-eg")
    enter_box.click()

    return driver

def place_buy_order(stock, quantity):
    driver = _log_in()
    driver.quit()

def place_sell_order(stock, quantity):
    driver = _log_in()
    driver.quit()
