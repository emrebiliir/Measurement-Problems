############################################
# SORTING REVIEWS
############################################
# Bazı yanlış sıralamalar üzerinden doğru sıralamanın nasıl yapılacağını ele alacağım.

# İlgili kütüphaneleri programıma dahil ediyorum.
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



# Uygulama 1 - Up-Down Diff Score = (up ratings) − (down ratings)

# İlk yöntem up down diference yöntemidir. Yani up rate' ler ile down rate'lerin farkını alıp buna göre bir sıralama yapmak.


# Senaryo; elimde iki tane review(yorum) var. Puanları binary yani ikili durumda olan yorumlarım var. Birincisi 600 like, 400 dislike almış.
#          İkincisinde de 5500 like, 4500 dislike var. Bunların her birisine uyarlanabilecek olan yöntemleri göstereceğim.

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

# Fonksiyonu yazıyoruz. Up ve down sayılarını verdiğimde bunların farkını döndürüyor.
def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)            # 200

# Review 2 Score
score_up_down_diff(5500, 4500)          # 1000

# Değerlendirme -----> Bu sonuçlara göre yüksek ihtimalle review 2' yi yukarıda tutardık. Fakat bir problem var. Review' ların  %si nedir diye baktığımızda  1.review' un
#                      %60, 2. yorumun yüzdeki ise %55' tir. Dolayısıyla farklılıktan dolayı sanki 2.si kazanıyor gibi gözükse de aslında %likten dolayı 1.si kazanmaktadır.
#                      Ama 2.si daha fazla oy almış. Burada bir threshold yani eşik değer olması gerekir. Aslında ikinciden gelen %55 değeri burada 550 450 değeri olsa da aynı
#                      değer gelecektir. Burdaki frekans bilgisi önemli fakat % lik olarak yoruma gelen bilgisi eksiktir. Dolayısıyla updown_d_s yöntemiyle 2 tane değer
#                      üzerinden böyle bir fark işlemine göre bu işlemi yapmak yeteri kadar doğru olmamaktadır.



# Uygulama 2 - # Score = Average rating = (up ratings) / (all ratings)

# Fonsiyonu yazıyorum.
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)                   # 0.6
score_average_rating(5500, 4500)                 # 0.55

# Bu yöntem daha mantıklı gelmektedir.
# Aşağıdaki senaryoyu da ele alayım. Yöntemin çalışılırlığına bakayım.

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)                       # 1
score_average_rating(100, 1)                     # 0.99

# Değerlendirme -----> Score_average_rating yöntemiyle bir sıralama yapıldığında 1. si yukarıda olacaktır. Peki gerçekten kazanması gereken 1.si midir?? Sayı yüksekliğini,
#                      frekans yüksekliğini göz önünde bulunduramadı. Dolayısıyla hem oran bilgisi hem frekans bilgisi eş zamanlı olarak göz önünde bulundurulabilecek
#                      şekilde bir sıralama skoru elde etmem gerekmektedir.



# Uygulama 3 - # Wilson Lower Bound Score (WLB Score)

# Bize ikili interaction' lar barındıran herhangi bir item, product ya da review' u skorlama imkanı sağlar. Örnek olarak youtubetaki videolar. Onlarda like/dislike, yorumları
# düşünelim. Buna benzer şekilde ikili etkileşimler sonucu ortaya çıkan bütün ölçme problemlerinde bize yardımı dokunur.

# Peki WLB skor teknik olarak şunu yapar; bernoulli parametresi olan p için bir güven aralığı hesaplar ve bu güven aralığının alt sınırını wlb skor olarak kabul eder.
# Bernoulli bir olasılık dağılımıdır. İkili olayların olasılığını hesaplamak için kullanılır. İki sonucu olan herhangi bir olayın nasıl gerçekleşebileceği olasılığını, örneğin
# bir dönüşüm oranı metriği gibi bir metriğe göre düşünüldüğünde bu dönüşüm oranına ilişkin değerlendirmenin gerçekleşmesi olasılığı gibi ikili olayların gerçekleşmesi olasılığını
# hesaplamak için kullanılan bir olasılık dağılımıdır. Bir olayın gerçekleşmesi olasılığı diye ifade ettiğim şey up olayıdır. Yani 'up/ all' olay dediğimizde burdaki orana
# ilişkin bir güven aralığı verir. Güven aralığı gibi olasılıksal işlemlere girilmesinin sebebi; örneğin elimizde müşterilerimizle ilgili bütün etkileşimlerin olmadığı bir
# senaryo var. Diyelim ki bir kullanıcı bir yorum yaptı. Bu yoruma beğendi beğenmedi gibi bazı etkileşimler geldi. Bazı etkileşimler geldi ama gelebilecek olası bütün etkileşimleri
# bilmiyorum. Bütün veri elimde yok. Ama elimde bir örneklem var örneğin 600 like 400 tane dislike. Dolayısıyla var olanların içerisinden bir up oranım var aslında. Şimdi
# buradan öyle bir genelleme yapmak istiyorum ki bilimsel olsun, bunu bütün kitleye yansıtabileyim ve bana güvenilir bir referans noktası versin. Dolayısıyla bu problemi bir
# olasılık problemi olarak ele aldığımızda ve bu ilgilendiğimiz olayı ifade eden oran değeri üzerinden bir güven değeri aralığı hesapladığımızda elimizde çok değerli şöyle bir
# bir bilgi olacak; örneğin up sayımız 600 down sayımız 400. Burada up oranımız 0.6' dır. Biz bu 0.6 ifadesi için bir güven aralığı hesapladığımızda 0.5 ile 0.7 ifadeleri gibi
# bazı aralıklar belirleriz ve artık elimde şu yorum olur; "100 kullanıcıdan 95' i bu yorumla ilgili bir etkileşim sağladığında %5 yanılma payım olmakla birlikte bu yorumun up
# oranı 0.5 ile 0.7 arasında olacaktır" diyorum". Alt sınır olan 0.5' i skor olarak belirliyorum.  Dolayısıyla bunu bütün gözlem birimleri için yaptığımda her birisi için
# garanti bir alt skorum var. Bu garanti alt skorumu referans alarak bunları sıralayabilirim.

# Fonksiyonu yazıyorum.
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)                # 0.56
wilson_lower_bound(5500, 4500)              # 0.54

wilson_lower_bound(2, 0)                    # 0.34
wilson_lower_bound(100, 1)                  # 0.94


# Değerlendirme ------> Wlb şunu sağlıyor; aslında sosyal kanıta bakarsak 2.si daha iyi ama aslında oransal olarak 1.si daha iyidir. Sonuçlar faydalı bulunma skorudur.
#                       demek ki 2.si 1.si kadar faydalı bulunmamış demektir.



# Case Study

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]

comments = pd.DataFrame({"up": up, "down": down})

# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

comments.head()





























