from bs4 import BeautifulSoup
import requests

resp = requests.get("https://daringtolivefully.com/simple-life-rules").text

soup = BeautifulSoup(resp, "html.parser")
main_div = soup.find("div", class_="post_content")
paragraphs = soup.find_all("p")
paragraphs = [(paragraph.text.split(". ")[1] + "\n") for paragraph in paragraphs[3:77]]
for paragraph in paragraphs:
    if "�" in str(paragraph):
        print(True)
    else:
        print(False)
with open("lessons.txt", "w") as f:
    f.writelines(paragraphs)
