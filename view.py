# __author__ = 'xiongjiaheng'
# -*- coding:utf-8 -*-
import tornado.ioloop
import tornado.web
import tornado.httpserver
import os
from cnn_keras_predict import transimg, predict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tornado.options import define, options, parse_command_line

define('port', default=8210, type=int)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html ')

    def post(self, *args, **kwargs):

        file_metas = self.request.files["pic"]
        # print(file_metas)
        for meta in file_metas:
            file_name = meta['filename']

            with open('./images/' + file_name, 'wb') as up:
                up.write(meta['body'])

        img_path = './images/' + file_name

        if transimg(img_path) is not False:
            img_path = str(transimg(img_path))
            lena = mpimg.imread(img_path)  # 读取和代码处于同一目录下的 lena.png
            #  此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
            plt.imshow(lena)  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.show()
        else:
            self.write("error")
        result = predict(img_path)
        self.render('result.html', result=result)


class ReturnHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('index.html ')


def make_app():
    return tornado.web.Application(handlers=[
        (r'/', MainHandler),
        (r'/result', ReturnHandler),
    ],
        template_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
        static_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
        cookie_secret='agdfiuwetr9w4689rfhjdc'
    )


if __name__ == '__main__':
    parse_command_line()
    app = make_app()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
