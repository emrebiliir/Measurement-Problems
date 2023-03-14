###################################################
# Sorting Products
###################################################
# Gerçek bir veri seti üzerinden bazı ürünlerin sıralamasının nasıl gerçekleştirilebileceğini değerlendireceğim.

# Uygulama: Kurs Sıralama

# Değişkenler;

# course_name: Kurs İsmi
# instructor_name: Eğitmen Adı
# purchase_count: Satın Alınma Sayısı
# rating: Genel Puan
# comment_count: Yapılan Yorum Sayısı
# 5_point: Toplam 5 Puan Sayısı
# 4_point:Toplam 4 Puan Sayısı
# 3_point:Toplam 3 Puan Sayısı
# 2_point:Toplam 2 Puan Sayısı
# 2_point:Toplam 1 Puan Sayısı


# İlgili kütüphaneleri programa dahil ediyorum ve csv dosyasını okuyorum.
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/product_sorting.csv")
df.head(10)



# Uygulama 1 - Sorting by Rating (Genel Puan İle Sıralama)

df.sort_values("rating", ascending=False).head(10)

# Değerlendirme ------>  1. satırın satın alma sayısı, genel puan ve toplam puanlara bakıldığında mantıklı gibi gözüküyor fakat 2. satır puan (rating), yorum ve satın alma
#                        açısından mantıklı gibi ama bu ilgili arama neticesinde asıl odağım olmayan bir kurs olabilir. Bir 'veri bilimi' anahtar kelimesiyle arama yapıldığı
#                        varsayımımız var. Dolayısıyla tam olarak karşılar mı karşılamaz mı bilmiyorum deyip burda bir düşünüyorum. Başka problemlerimiz var mı diye incelemeye
#                        devam ediyorum. 4. satırda bir problem daha dikkatimizi çekiyor. Rating' i çok yüksek ama bu kursun yorum sayısı diğerlerine kıyasla oldukça düşük.
#                        Diğer yandan daha aşağılarda daha fazla satın alma sayısına sahip başka kurslar var bunlar olduğu halde bir şekilde satın alma sayısı ve yorum sayısı
#                        rating' in altında ezilmiş. Bu faktörleri göz ardı edemem çünkü sosyal ispat olayının kullanıcılar üzerinde çok büyük bir etkisi vardır. Sonuç olarak
#                        sadece rating' e göre sıralamak işimi çözmeyecektir. Bazı göz ardı edilemeyecek durumları kapsamadığından dolayı mantıklı değildir. Dolayısıyla hem
#                        satın alma sayısını hem puanını hem de yorum sayısını aynı anda göz önünde bulundurmalıyım.



# Uygulama 2 - Sorting by Comment Count or Purchase Count (Yorum Sayısına ya da Satın Alma Sayısına Göre Sıralama)

# Sıralama işlemini yorum sayısı ve satın alma sayısına göre bir değerlendirip hala tek başına da bunlara göre olamayacağı fikrini derinleştirelim.
# Toplam puana ya da satın alma sayısına göre de sıralayabiliriz.

# Toplam Satın almaya göre sıralama;
df.sort_values("purchase_count", ascending=False).head(20)

# Değerlendirme ------> Yeni bir problem dikkatimi çekiyor. Sadece satın alınma sayısına göre sıraladığımda yorumu az olanlar ve düşük kurs puanına sahip olanlar da geldi.
# Belki de 11 indexli olan ücretsiz açılan bir kurs olabilir. Bu yüzden bir çok insan kaydoldu. Bundan dolayı satın alması yüksek olabilir. Netice itibariyle insanlar kursa
# girip gözlemlediğinde bunun rating' ini, kendisi için faydasını bu şekilde puanladı. Dolayısıyla bu kursun bu kadar yukarda olmaması gerektiği fikri de oldukça akla yatkındır.

# Yorum sayısına göre sıralama;
df.sort_values("commment_count", ascending=False).head(20)

# Değerlendirme ------> Yine bazı yüksek satın alma ya da yüksek yorumlanma değerlerine sahip kurslar yukarıya geldi. Bir şekilde sosyal ispatı en sonda yorum sayısına göre
#                       odakladığımda sanki diğerlerine göre biraz daha iyi bir sonuç geldi. Fakat burada yine benzer problemler var. Tek başına yorum sayısına göre bir şey
#                       yapılmamalı. Çünkü düşük puanlı olanlar da var. Bir şekilde satın alması yüksek ama bunlar ücretsiz bir şekilde dağıtılmış olabilir. Dolayısıyla sadece
#                       ratinge göre sıraladığımızda diğer önemli olabilecek faktörlerin etkisi ezilmektedir. Öyle bir şey yapmalıyım ki benim için önemli olabilecek bu 2
#                       faktörü de  aynı anda göz önünde bulundurabileyim. Bunu yapmanın yolu bu 3 faktörü belirli bir standart çizgiye çekip (standartlaştırıp) daha sonra
#                       tek bir skor (tek bir metrik) haline getirip ondan sonra bütün kursları sıralamaktır.



# Uygulama 3 - Sorting by Rating, Comment and Purchase (Puan, Yorum ve Satın Almaya Göre Sıralama)

# Bu aşamada kütüphanemize minmaxscaler metodunu ekliyoruz. Bu işlemi en başta yaptım. Bu metod metrikleri standartlaştırmak için kullanılır. Akla ilk gelen şey aslında bu 3
# değeri birbiriyle çarpıp bir skor oluşturmaktır. Fakat satın alma sayıların ölçekleri daha geniş, daha büyük sayılardan oluşuyor. Benzer şekilde yorum sayıları da büyük
# sayılardan oluşuyor. Fakat rating çok düşük kalıyor. O zaman bu üçünü çarptığımızda rating her türlü ezilir. Yorum açısından bir şekilde üç haneye gelememiş kurslar da ezilir.
# Gözlemleneceği üzere bu değerleri olduğu gibi çarpmak doğru değildir. Bunun için minmaxscaler metodunu kullanıp hepsini aynı ölçeğe getiriyorum. Örneğin rating' in cinsinden
# ifade edeceğim. Rating 1 ile 5 arasındaki sayılardan oluşuyor. Dolayısıyla diğer değişkenleri de 1 ile 5 arasındaki sayılara dönüştürüyorum.

df["purchase_count_scaler"] = MinMaxScaler(feature_range= (1,5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])
df["comment_count_scaler"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["commment_count"]]).transform(df[["commment_count"]])

# Değerlendirme -----> Yeni bir değişken oluşturuyorum. MinMaxScaler metoduyle standartlaştırma işleminde skor değerlerinin 1 ile 5 arasında olmasını sağladım. fit metoduyla
#                      da dönüştürmeyi işlemini yapıyorum. DÖnüştürme işleminden sonra elimde eski değişken olan purchase_count ve oluşturduğum bu yeni değişken olan
#                      purchase_count_scaler değişkeni var. transform metoduyla da yeni değiştirdiğim değerlere bunu dönüştür demiş oluyorum.


# İlgilendiğim değişkenler aynı cinsten olduğundan bunların ortalamasını alınabilir ya da ağırlıklandırılabilinir. O halde burada bir hassaslaştırma daha yapmak istiyorum.
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaler"] * w1 / 100 +
            dataframe["purchase_count_scaler"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

# Faktörlerin ağırlıklarıyla oluşturduğum skorlar üzerinden bir sıralama yapıyorum.
df.sort_values("weighted_sorting_score", ascending=False).head(20)

# Değerlendirme ------> Önceden yukarıda görmediğim kursları görebiliyorum. Artık çok daha güvenilir bir durumda benim için.


# Burada ilgisiz olan kursları da çıkarıyorum.
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)



# Uygulama 4 - Bayesian Average Rating Score

# Elimde geçmişteki elde ettiğim puanların dağılımları var. Yani geçmiş bir bilgi var. Bayesian, geçmiş bilgiyi kullanarak gelecekle ilgili bir şeyler yapma yaklaşımıyla
# elimdeki bu var olan değerler üzerinden tekrar bir rating hesabı yapar. Bir önceki işlemlerden farkı olayın içerisinde istatistik ve algoritmalar katacak olmamdır.
# Yapacak olduğum şey; puanların (5_p, 4_p....) dağılımıdır. Bu puanların dağılım bilgisini kullanarak olasılıksal bir ortalama hesaplayacağım. Bayesian Average Rating Score,
# puan dağılımlarının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesabı yapar. Daha sonra dilersem buna göre sıralayabilirim.
# Bu fonksiyon için math ve scipy kütüphanelerini kullanıyor olacağım.

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

# Değerlendirme -----> n ifadesi girilecek olan yıldızların ve bu yıldızlara ait gözlem frekanslarını ifade etmektedir. 1 puanlı yıldız ile başlayıp 5 puanlı yıldız ile
#                      sonlandırıyorum. Şimdi sadece ilgili 5 değişkeni seçip 5 değişkeni de n argümanına ters bir sırada göndereceğim.


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",                  # axis=1 diyerek sütunlarda bir işlem gerçekleştireceğimi söyledim.
                                                                "2_point",                  # Değişkenleri tersten verdim.
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)
# Değerlendirme -----> Bu yöntemle gözden kaçırilan bazı durumlar söz konusudur. Yine yorum sayıları, satın alma sayıları gözden kaçtı. Eğer sadece tek odağım verilen puanlar
#                      olsaydi ve bu puanlara göre bir sıralama yapmak istiyor olsaydım bu durumda bayesian yöntemi tek başına kullanılabilirdi. Fakat bu durumda
#                      bu çok da geçerli olmayacaktır.



# Uygulama 5 - Hybrid Sorting (BAR Score + Diğer Faktorler)

# Olasılıksal olarak bu bar scor fonksiyonu ile diğer faktörleri birleştirdiğim weighted_sorting_score fonksiyonunu birleştiriyorum.
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# Değerlendirme -----> Course9' un çıktıda olması oldukça değerli. Puanının düşük olduğuna bakmayın. Kaydadeğer bir yorum sayısı ve satın alması var.
# Daha güzel olan şey Course1' in de gördündüğü konumda olmasıdır. Belli ki bu kurs yeni bir kurs. Puanı epey yüksek satın alması da iyi durumda görünüyor. Fakat yorum sayısı
# baya düşük. Bir potansiyel vaad ediyor ki bu liste içerisinde bulunuyor. Bu potansiyeli fonksiyonda %60 ağırlık verdiğimiz bar_score ile yakaladık. bar_score bana veri seti
# içerisinde yeni olsa da potansiyel vaad edenleri de yukarıya taşıma şansı sağadı.


# Özetle; bayesian yöntemi hibrit bir sıralamada ağırlığı olan bir faktör olarak göz önünde bulundurulduğunda bir şekilde potansiyeli daha yüksek ama henüz yeterli sosyal
# kanıt'ı alamamış ürünleri de yukarı çıkarmaktadır.






