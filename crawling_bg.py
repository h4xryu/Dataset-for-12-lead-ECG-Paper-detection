def crawl_image_data(key_word:str, max_num:int, des_dir:str):
    from icrawler.builtin import GoogleImageCrawler
    google_crawler = GoogleImageCrawler(storage={'root_dir': des_dir})
    google_crawler.crawl(keyword=key_word, max_num=max_num)


if __name__ == '__main__':
    key_word = 'top view table background'; max_num = 10000; des_dir = './bg_noises'
    crawl_image_data(key_word, max_num, des_dir)