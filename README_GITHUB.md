# ₿ Bitcoin Analysis Dashboard

**Real-time Bitcoin analysis with ML predictions, trading strategies, and market insights**

[![Auto-Update](https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/actions/workflows/update-dashboard.yml/badge.svg)](https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/actions/workflows/update-dashboard.yml)
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-success)](https://bitcoin-analysis-dashboard.vercel.app)

---

## 🌐 Live Dashboard

**👉 [View Live Dashboard](https://bitcoin-analysis-dashboard.vercel.app)**

Updated automatically every 10-15 minutes with fresh Bitcoin data!

---

## ✨ Features

### 📊 **Multi-Tab Analysis Interface**

- **Overview** - Current price, key levels, technical indicators
- **ML Prediction** - Machine learning price forecasts
- **Trading Strategy** - 50-week SMA signals with risk management
- **M2 Analysis** - Money supply correlation with Bitcoin
- **Lead-Lag Analysis** - Economic indicator relationships

### 🤖 **Fully Automated**

- ✅ Updates every 10-15 minutes
- ✅ Fetches live Bitcoin prices
- ✅ Regenerates ML predictions
- ✅ Updates trading signals
- ✅ Auto-deploys to Vercel
- ✅ **Runs 24/7 in the cloud (GitHub Actions)**

### 📈 **Comprehensive Data**

- Bitcoin OHLCV data (10+ years)
- M2 money supply analysis
- Federal Reserve interest rates
- Technical indicators (50+ indicators)
- Market phase classification
- Volatility analysis

### 🎨 **Professional UI**

- ₿ Bitcoin logo favicon
- Responsive design
- Interactive tabs
- Embedded charts (no external dependencies)
- Auto-refresh every 5 minutes
- Clean, modern interface

---

## 🏗️ Architecture

```
GitHub Actions (Cloud)
    ↓
Fetch Bitcoin Data (yfinance)
    ↓
Generate ML Analysis
    ↓
Generate Trading Signals
    ↓
Create Dashboard HTML
    ↓
Commit to GitHub Repo
    ↓
Deploy to Vercel
    ↓
🌐 Live Dashboard Updated!
```

**Benefits:**
- No server needed
- Completely free
- Runs 24/7 automatically
- Version-controlled data
- Full audit trail

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitHub account
- Vercel account

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard.git
cd bitcoin-analysis-dashboard/ml_pipeline

# Create virtual environment
cd ..
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy yfinance requests matplotlib seaborn scikit-learn

# Generate dashboard
cd ml_pipeline
python3 generate_comprehensive_bitcoin_website.py

# View locally
cd reports
python3 -m http.server 8080
# Visit: http://localhost:8080
```

### Cloud Deployment

See **[GITHUB_AUTO_UPDATE_SETUP.md](GITHUB_AUTO_UPDATE_SETUP.md)** for complete setup guide.

**Quick summary:**
1. Create GitHub repository
2. Add Vercel credentials as GitHub secrets
3. Push code with GitHub Actions workflow
4. Auto-updates start immediately!

---

## 📁 Project Structure

```
bitcoin-analysis-dashboard/
├── .github/
│   └── workflows/
│       └── update-dashboard.yml    # Auto-update workflow
├── ml_pipeline/
│   ├── datasets/
│   │   └── btc-ohlc.csv           # Bitcoin price data
│   ├── reports/
│   │   ├── index.html             # Main dashboard
│   │   ├── *.json                 # Analysis results
│   │   └── *.html                 # Other reports
│   ├── fetch_live_bitcoin_data.py
│   ├── generate_m2_interest_rate_bitcoin_study.py
│   ├── generate_trading_strategy_analysis.py
│   └── generate_comprehensive_bitcoin_website.py
├── requirements.txt
├── README.md
└── GITHUB_AUTO_UPDATE_SETUP.md
```

---

## 🔧 Technologies Used

### Backend
- **Python 3.11** - Main language
- **pandas** - Data manipulation
- **yfinance** - Bitcoin price data
- **scikit-learn** - ML models
- **matplotlib** - Chart generation

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling (no frameworks!)
- **Vanilla JavaScript** - Interactivity
- **Base64 embedded charts** - No external dependencies

### Infrastructure
- **GitHub Actions** - Automated workflows
- **Vercel** - Static site hosting
- **GitHub** - Code & data storage

---

## 📊 Data Sources

- **Bitcoin Price:** Yahoo Finance (yfinance)
- **M2 Money Supply:** FRED API (Federal Reserve Economic Data)
- **Interest Rates:** FRED API
- **Technical Indicators:** Calculated from OHLCV data

---

## 🤝 Contributing

Contributions welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📈 Future Enhancements

- [ ] Multi-cryptocurrency support (ETH, SOL, etc.)
- [ ] Email alerts for price movements
- [ ] Dark mode toggle
- [ ] Mobile app
- [ ] Twitter bot integration
- [ ] Historical data export
- [ ] More ML models (LSTM, Transformer)
- [ ] Sentiment analysis from social media

---

## 📝 License

MIT License - feel free to use this for your own projects!

---

## 🙏 Acknowledgments

- Data from [Yahoo Finance](https://finance.yahoo.com)
- Economic data from [FRED](https://fred.stlouisfed.org)
- Hosted on [Vercel](https://vercel.com)
- Automated with [GitHub Actions](https://github.com/features/actions)

---

## 📞 Contact

Questions? Feedback? Open an issue or reach out!

---

## ⭐ Star this repo if you find it useful!

**Built with ❤️ and lots of ☕**

---

**Current Bitcoin Price:** Check the [live dashboard](https://bitcoin-analysis-dashboard.vercel.app) for real-time data!
