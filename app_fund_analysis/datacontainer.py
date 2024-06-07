class DataContainer:
    """
    Contains all the xpaths and urls and others variables that may change from time to time
    """

    # Urls
    BASE_URL_ZONE_BOURSE = "https://www.google.com/search?q=zone+bourse+{}+finance&sxsrf=ALiCzsbIaWNWrnXJ5acLqlPx2kINT72YMA%3A1670610120483&ei=yHyTY9CSHcmPkdUP3veM2AI&oq=zone+bourse+telenor++finance&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgQIIxAnOggIABCiBBCwA0oECEEYAUoECEYYAFCMBViMBWCHEGgBcAB4AIABKYgBKZIBATGYAQCgAQHIAQPAAQE&sclient=gws-wiz-serp"
    BASE_URL_FINVIZ = "https://finviz.com/quote.ashx?t={}&p=d"

    # Zone bourse
    GET_URL_CAPCHA_XPATH = "(//button)[4]"
    GET_URL_XPATH = '(//div[contains(@data-async-context , "query:zone")]/div//a)[1]'
    GET_DESCRIPTION_XPATH = '//div[@class="company-logo"]/following-sibling::text()'

    # Finviz
    FINVIZ_POPUP_XPATH = '//*[@id="qc-cmp2-ui"]/div[2]/div/button[3]'
