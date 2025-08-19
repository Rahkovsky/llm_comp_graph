from sec_downloader import Downloader

# Initialize the downloader with your company name and email
dl = Downloader("MyCompanyName", "email@example.com")


# Download the latest 10-Q filing for Apple
html = dl.get_filing_html(ticker="AAPL", form="10-Q")

html