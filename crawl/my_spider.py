import scrapy

class BlogSpider(scrapy.Spider):
    name = 'cse-mit'
    start_urls = ['https://cse.mit.edu/programs/']

    def parse(self, response):
        for link in response.css('a::attr(href)').getall():
            print(link)
    # def parse(self, response):
    #     for title in response.css('.oxy-post-title'):
    #         yield {'title': title.css('::text').get()}

    #     for next_page in response.css('a.next'):
    #         yield response.follow(next_page, self.parse)
