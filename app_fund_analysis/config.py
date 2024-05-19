BENCHMARK_TICKERS = (
    "KO",
    "JNJ",
    "XOM",
    "MMM",
    "ITW",
    "IBM",
    "O",
    "PG",
    "EPD",
    "BLK",
    "VZ",
    "NWN",
    "HD",
    "LEG",
)

ticker_suffix_to_currency = {
    "A": "AUD",  # NYSE ARCA - Australian Dollar
    "AX": "AUD",  # Australian Securities Exchange (ASX) - Australian Dollar
    "BA": "ARS",  # Buenos Aires Stock Exchange (BYMA) - Argentine Peso
    "BC": "CAD",  # Toronto Stock Exchange (TSX) - Canadian Dollar
    "BD": "BHD",  # Bahrain Bourse (BHB) - Bahraini Dinar
    "BE": "EUR",  # Euronext Brussels - Euro
    "BK": "THB",  # Stock Exchange of Thailand (SET) - Thai Baht
    "BM": "BMD",  # Bermuda Stock Exchange (BSX) - Bermudian Dollar
    "BN": "BOB",  # Bolsa Boliviana de Valores (BBV) - Bolivian Boliviano
    "BO": "BRL",  # B3 - Brasil Bolsa Balcão - Brazilian Real
    "BR": "BRL",  # Euronext Brussels - Brazilian Real
    "BZ": "BZD",  # Belize Stock Exchange (BZSE) - Belize Dollar
    "CA": "CAD",  # Canadian Securities Exchange (CSE) - Canadian Dollar
    "CN": "CNY",  # Shanghai Stock Exchange (SSE) - Chinese Yuan
    "CO": "COP",  # Bolsa de Valores de Colombia (BVC) - Colombian Peso
    "CR": "CRC",  # Bolsa Nacional de Valores (BNV) - Costa Rican Colon
    "CT": "CAD",  # Canadian Securities Exchange (CSE) - Canadian Dollar
    "CX": "CHF",  # SIX Swiss Exchange - Swiss Franc
    "CY": "CYP",  # Cyprus Stock Exchange (CSE) - Cyprus Pound
    "DE": "EUR",  # XETRA - Euro
    "DL": "DKK",  # OMX Nordic Exchange Copenhagen - Danish Krone
    "DR": "DOP",  # Bolsa de Valores de la República Dominicana (BVRD) - Dominican Peso
    "DU": "EUR",  # Euronext Dublin - Euro
    "DV": "EUR",  # Euronext Dublin - Euro
    "DY": "EGP",  # Egyptian Exchange (EGX) - Egyptian Pound
    "EC": "EUR",  # Euronext Brussels - Euro
    "EL": "EUR",  # Euronext Lisbon - Euro
    "EM": "EUR",  # Euronext Amsterdam - Euro
    "EP": "EUR",  # Euronext Paris - Euro
    "ES": "EUR",  # Euronext Brussels - Euro
    "F": "EUR",  # Euronext Paris - Euro
    "FI": "EUR",  # NASDAQ OMX Helsinki - Euro
    "FR": "EUR",  # Euronext Paris - Euro
    "HA": "EUR",  # NASDAQ OMX Helsinki - Euro
    "HE": "EUR",  # Euronext Amsterdam - Euro
    "HK": "HKD",  # Hong Kong Stock Exchange (HKEX) - Hong Kong Dollar
    "HM": "HNL",  # Honduras Stock Exchange (BHV) - Honduran Lempira
    "HN": "HNL",  # Honduras Stock Exchange (BHV) - Honduran Lempira
    "I": "EUR",  # Borsa Italiana (Milan Stock Exchange) - Euro
    "IC": "ISK",  # OMX Nordic Exchange Iceland - Icelandic Krona
    "IL": "ILS",  # Tel Aviv Stock Exchange (TASE) - Israeli Shekel
    "IS": "ILS",  # Tel Aviv Stock Exchange (TASE) - Israeli Shekel
    "IT": "EUR",  # Borsa Italiana (Milan Stock Exchange) - Euro
    "JK": "IDR",  # Indonesia Stock Exchange (IDX) - Indonesian Rupiah
    "JO": "JOD",  # Amman Stock Exchange (ASE) - Jordanian Dinar
    "JS": "JMD",  # Jamaica Stock Exchange (JSE) - Jamaican Dollar
    "JV": "JMD",  # Jamaica Stock Exchange (JSE) - Jamaican Dollar
    "K": "LAK",  # Lao Securities Exchange (LSX) - Lao Kip
    "KA": "KWD",  # Kuwait Stock Exchange (KSE) - Kuwaiti Dinar
    "KS": "KRW",  # Korea Stock Exchange (KRX) - South Korean Won
    "KQ": "KRW",  # KOSDAQ - South Korean Won
    "L": "GBP",  # London Stock Exchange (LSE) - British Pound
    "LA": "LAK",  # Lao Securities Exchange (LSX) - Lao Kip
    "LG": "LVL",  # NASDAQ OMX Riga - Latvian Lats
    "LI": "LTL",  # NASDAQ OMX Vilnius - Lithuanian Litas
    "LN": "EUR",  # London Stock Exchange (LSE) - Euro
    "LS": "LBP",  # Beirut Stock Exchange (BSE) - Lebanese Pound
    "LT": "EUR",  # NASDAQ OMX Vilnius - Euro
    "LV": "EUR",  # NASDAQ OMX Riga - Euro
    "LZ": "LSL",  # Lesotho Stock Exchange (LSE) - Loti
    "M": "MYR",  # Bursa Malaysia (MYX) - Malaysian Ringgit
    "MA": "MAD",  # Casablanca Stock Exchange (CSE) - Moroccan Dirham
    "MC": "EUR",  # Euronext Amsterdam - Euro
    "MD": "MKD",  # Macedonian Stock Exchange (MSE) - Macedonian Denar
    "ME": "EUR",  # Euronext Amsterdam - Euro
    "MF": "EUR",  # Euronext Paris - Euro
    "MI": "EUR",  # Borsa Italiana (Milan Stock Exchange) - Euro
    "MK": "EUR",  # Euronext Amsterdam - Euro
    "MM": "MZN",  # Bolsa de Valores de Moçambique (BVM) - Mozambican Metical
    "MO": "EUR",  # Euronext Amsterdam - Euro
    "MP": "EUR",  # Euronext Paris - Euro
    "MR": "MUR",  # Stock Exchange of Mauritius (SEM) - Mauritian Rupee
    "MS": "MUR",  # Stock Exchange of Mauritius (SEM) - Mauritian Rupee
    "MT": "EUR",  # Malta Stock Exchange (MSE) - Euro
    "MU": "EUR",  # Euronext Amsterdam - Euro
    "MV": "MVR",  # Maldives Stock Exchange (MSE) - Maldivian Rufiyaa
    "MW": "MWK",  # Malawi Stock Exchange (MSE) - Malawian Kwacha
    "MX": "MYR",  # Bursa Malaysia (MYX) - Malaysian Ringgit
    "NA": "NAD",  # Namibian Stock Exchange (NSX) - Namibian Dollar
    "NB": "NOK",  # Oslo Stock Exchange (OSE) - Norwegian Krone
    "NC": "XPF",  # NYSE Euronext (Paris) - CFP Franc
    "ND": "NAD",  # Namibian Stock Exchange (NSX) - Namibian Dollar
    "NE": "NGN",  # Nigerian Stock Exchange (NSE) - Nigerian Naira
    "NL": "NIO",  # Bolsa de Valores de Nicaragua (BVN) - Nicaraguan Cordoba
    "NM": "NPR",  # Nepal Stock Exchange (NEPSE) - Nepalese Rupee
    "NS": "INR",  # National Stock Exchange - Indian Rupee
    "NT": "TWD",  # Taiwan Stock Exchange (TWSE) - New Taiwan Dollar
    "NX": "NZD",  # New Zealand Exchange (NZX) - New Zealand Dollar
    "NZ": "NZD",  # New Zealand Exchange (NZX) - New Zealand Dollar
    "OL": "NOK",  # Oslo Stock Exchange (OSE) - Norwegian Krone
    "OM": "OMR",  # Muscat Securities Market (MSM) - Omani Rial
    "PA": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PB": "USD",  # Bolsa Electrónica de Valores (BEV) - US Dollar
    "PC": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PD": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PE": "PEN",  # Lima Stock Exchange (BVL) - Peruvian Sol
    "PG": "PGK",  # Port Moresby Stock Exchange (POMSoX) - Papua New Guinean Kina
    "PH": "PHP",  # Philippine Stock Exchange (PSE) - Philippine Peso
    "PK": "PKR",  # Pakistan Stock Exchange (PSX) - Pakistani Rupee
    "PL": "PLN",  # Warsaw Stock Exchange (GPW) - Polish Zloty
    "PN": "PYG",  # Bolsa de Valores y Productos de Asunción (BVPASA) - Paraguayan Guarani
    "PR": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PS": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PT": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PV": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PW": "USD",  # NYSE Euronext (Paris) - US Dollar
    "PX": "USD",  # NYSE Euronext (Paris) - US Dollar
    "Q": "USD",  # NASDAQ Stock Market (NASDAQ) - US Dollar
    "R": "BRL",  # BM&F Bovespa (BVMF) - Brazilian Real
    "RA": "ZAR",  # Johannesburg Stock Exchange (JSE) - South African Rand
    "RG": "BRL",  # BM&F Bovespa (BVMF) - Brazilian Real
    "RI": "IDR",  # Indonesia Stock Exchange (IDX) - Indonesian Rupiah
    "RM": "MYR",  # Bursa Malaysia (MYX) - Malaysian Ringgit
    "RN": "BRL",  # BM&F Bovespa (BVMF) - Brazilian Real
    "RO": "RON",  # Bucharest Stock Exchange (BVB) - Romanian Leu
    "RP": "PHP",  # Philippine Stock Exchange (PSE) - Philippine Peso
    "RR": "BRL",  # BM&F Bovespa (BVMF) - Brazilian Real
    "RT": "RUB",  # Moscow Exchange (MOEX) - Russian Ruble
    "SA": "ZAR",  # Johannesburg Stock Exchange (JSE) - South African Rand
    "SB": "ZAR",  # Johannesburg Stock Exchange (JSE) - South African Rand
    "SC": "SCR",  # Seychelles Stock Exchange (Trop-X) - Seychellois Rupee
    "SD": "SAR",  # Saudi Stock Exchange (Tadawul) - Saudi Riyal
    "SE": "SGD",  # Singapore Exchange (SGX) - Singapore Dollar
    "SG": "SGD",  # Singapore Exchange (SGX) - Singapore Dollar
    "SH": "CNY",  # Shanghai Stock Exchange (SSE) - Chinese Yuan
    "SI": "EUR",  # Euronext Amsterdam - Euro
    "SK": "KRW",  # Korea Stock Exchange (KRX) - South Korean Won
    "SL": "LKR",  # Colombo Stock Exchange (CSE) - Sri Lankan Rupee
    "SM": "SOS",  # Somalia Stock Exchange (SSE) - Somali Shilling
    "SN": "ZAR",  # Johannesburg Stock Exchange (JSE) - South African Rand
    "SO": "SOS",  # Somalia Stock Exchange (SSE) - Somali Shilling
    "SP": "USD",  # NYSE Euronext (Paris) - US Dollar
    "SR": "SRD",  # Suriname Stock Exchange (SSE) - Surinamese Dollar
    "SS": "ZAR",  # Johannesburg Stock Exchange (JSE) - South African Rand
    "ST": "EUR",  # Euronext Amsterdam - Euro
    "SU": "EUR",  # Euronext Amsterdam - Euro
    "SV": "EUR",  # Euronext Amsterdam - Euro
    "SW": "SEK",  # NASDAQ OMX Stockholm - Swedish Krona
    "SX": "EUR",  # Euronext Amsterdam - Euro
    "SY": "SYP",  # Damascus Securities Exchange (DSE) - Syrian Pound
    "SZ": "CNY",  # Shenzhen Stock Exchange (SZSE) - Chinese Yuan
    "TA": "TWD",  # Taiwan Stock Exchange (TWSE) - New Taiwan Dollar
    "TB": "THB",  # Stock Exchange of Thailand (SET) - Thai Baht
    "TC": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TD": "TND",  # Bourse des Valeurs Mobilières de Tunis (BVMT) - Tunisian Dinar
    "TE": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TF": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TG": "TTD",  # Trinidad and Tobago Stock Exchange (TTSE) - Trinidad and Tobago Dollar
    "TH": "THB",  # Stock Exchange of Thailand (SET) - Thai Baht
    "TI": "TTD",  # Trinidad and Tobago Stock Exchange (TTSE) - Trinidad and Tobago Dollar
    "TK": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TL": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TM": "TMT",  # Turkmenistan Securities Market (TSM) - Turkmenistan Manat
    "TN": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TO": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TP": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TR": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TT": "TTD",  # Trinidad and Tobago Stock Exchange (TTSE) - Trinidad and Tobago Dollar
    "TU": "TND",  # Bourse des Valeurs Mobilières de Tunis (BVMT) - Tunisian Dinar
    "TV": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TW": "TWD",  # Taiwan Stock Exchange (TWSE) - New Taiwan Dollar
    "TX": "TRY",  # Borsa Istanbul (BIST) - Turkish Lira
    "TZ": "TZS",  # Dar es Salaam Stock Exchange (DSE) - Tanzanian Shilling
    "UA": "UAH",  # Ukrainian Exchange (UX) - Ukrainian Hryvnia
    "UG": "UGX",  # Uganda Securities Exchange (USE) - Ugandan Shilling
    "UK": "GBP",  # London Stock Exchange (LSE) - British Pound
    "UL": "EUR",  # Euronext Amsterdam - Euro
    "UM": "USD",  # US Dollar
    "UN": "USD",  # US Dollar
    "UR": "UYU",  # Montevideo Stock Exchange (BVMB) - Uruguayan Peso
    "US": "USD",  # US Dollar
    "UV": "UZS",  # Uzbekistan Stock Exchange (UZSE) - Uzbekistan Som
    "UY": "UYU",  # Montevideo Stock Exchange (BVMB) - Uruguayan Peso
    "UZ": "UZS",  # Uzbekistan Stock Exchange (UZSE) - Uzbekistan Som
    "VA": "EUR",  # Euronext Amsterdam - Euro
    "VB": "EUR",  # Euronext Brussels - Euro
    "VI": "EUR",  # Euronext Amsterdam - Euro
    "VL": "EUR",  # Euronext Lisbon - Euro
    "VN": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VO": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VR": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VS": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VT": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VU": "VUV",  # Vanuatu Stock Exchange (VSE) - Vanuatu Vatu
    "VV": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "VX": "VND",  # Ho Chi Minh Stock Exchange (HOSE) - Vietnamese Dong
    "W": "USD",  # NYSE Euronext (Amsterdam) - US Dollar
    "WA": "WST",  # South Pacific Stock Exchange (SPSE) - Samoan Tala
    "WB": "USD",  # NYSE Euronext (Amsterdam) - US Dollar
    "WC": "USD",  # NYSE Euronext (Amsterdam) - US Dollar
    "WE": "EUR",  # Euronext Amsterdam - Euro
}
