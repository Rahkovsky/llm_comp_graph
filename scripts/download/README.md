# SEC Filing Download Scripts

Professional, modular, and efficient SEC filing download system.

## 🏗️ **Architecture**

### **Core Module (`core.py`)**
- **`SECDownloader`**: HTTP client with retry logic and rate limiting
- **`TextExtractor`**: Clean text extraction from SEC filings
- **`FilingIterator`**: Smart filtering of filings by date/type
- **`FileManager`**: Organized file path management

### **Download Scripts**
- **`download_10K.py`**: Download specific company 10-K filings


## 🚀 **Quick Start**

### **Single Company Download**
```bash
# Download Apple's latest 10-K
python scripts/download/download_10K.py --tickers AAPL --max-per-ticker 1

# Download multiple companies
python scripts/download/download_10K.py --tickers AAPL MSFT GOOGL --max-per-ticker 2

# Download from file
echo -e "AAPL\\nMSFT\\nGOOGL" > tickers.txt
python scripts/download/download_10K.py --tickers-file tickers.txt
```



## 📋 **Command Reference**

### **download_10K.py**
```
--tickers [TICKERS ...]       Ticker symbols
--tickers-file FILE           File with one ticker per line
--forms [FORMS ...]           Form types (default: 10-K, 10-K/A)
--start-date YYYY-MM-DD       Start date filter
--end-date YYYY-MM-DD         End date filter
--user-agent USER_AGENT       Custom User-Agent
--sleep SECONDS               Delay between requests (default: 0.25)
--max-per-ticker N            Max filings per ticker
```



## 🎯 **Features**

- ✅ **Professional**: Clean, modular code architecture
- ✅ **Efficient**: Retry logic, rate limiting, resume capability
- ✅ **Short**: Minimal, focused scripts
- ✅ **Tested**: Verified working with SEC API
- ✅ **Progress Tracking**: Real-time download progress
- ✅ **Text Extraction**: Automatic plain text extraction
- ✅ **Error Handling**: Robust error recovery

## 📁 **Output Structure**
```
data/input/10K/
├── AAPL/
│   └── 2024/
│       ├── AAPL_0000320193_2024-11-01_0000320193-24-000123.txt
│       └── AAPL_0000320193_2024-11-01_0000320193-24-000123_plain.txt
└── ALL_COMPANIES/
    └── 2024/
        ├── NVDA_0001045810_2024-02-21_0001045810-24-000029.txt
        └── MSFT_0000789019_2024-07-30_0000789019-24-000072.txt
```

## ⚡ **Performance**
- **Rate Limited**: 0.25s between requests (SEC compliant)
- **Parallel Processing**: Efficient batch operations
- **Resume Capability**: Continue interrupted downloads
- **Memory Efficient**: Streaming file downloads
