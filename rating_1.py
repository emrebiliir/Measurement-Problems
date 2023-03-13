########################################################################################################################
# RATING PRODUCTS
########################################################################################################################

# Amaç: Bir ürüne verilen puanlar üzerinden çeşitli değerlendirmeler yaparak en doğru puanın nasıl hesaplanabileceğine dair bir uygulama yapmak olacaktır.
#       Amacı gerçekleştirmek için farklı yöntemler kullanacağız.

# Yöntemler;
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


# Değişkenler;
# Rating - Kursun puanı
# Timestamp - Puanın verildiği tarih
# Enrolled - Üye olma tarihi
# Progress - Kursun izlenme oranı
# Questions Asked - Sorulan soru sayısı
# Questions Answered - Cevap verilen soru sayısı

# Senaryo;
# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

# 4.764925 şeklinde bir puan hesabı yapılmış. Bu puan hesabı üzerinden verinin kendisi elimde olduğundan dolayı bir değerlendirme yapacağız.

# İlgili kütüphaneleri yükleyip birkaç ayarlama yapıyoruz.
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Csv dosyasını okuyoruz.
df = pd.read_csv("datasets/course_reviews.csv")
df.head()

# Veriyi anlamak adına birkaç analiz yapıyoruz.

# Kursun ortalaması
df["Rating"].mean()
# Her bir rating' in adedi
df["Rating"].value_counts()
# Sorulan soru sayısına göre rating ortalamaları ve toplam soru sayısı
df.groupby("Questions Asked").agg({"Rating":"mean",
                                  "Questions Asked":"count"})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uygulama 1 - Average (Ortalama)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df["Rating"].mean()

# Yorum -----> Bu yöntem uygulandığında kullanıcıların son zamanlarındaki memnuniyet trend'leri gözden kaçırılıyor olabilir. Bir kullanıcı kursun önceki tarihlerinde kursu
#              almış olup beğenmemiş olabilir. Bir diğer kullanıcı ise en güncel tarihte kursu alıp çok memnun kalmış olabilir. Bu durumda belki de zamanla kendini geliştiren
#              bu kursun bu pozitif gelişimi göz ardı ediliyor olabilir.



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uygulama 2 - Time-Based Weighted Average (Zaman Ağırlıklı Ortalama)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Burada karşımızda bazı yapısal problemler oluşacaktır. Zamana göre bir hesap yapma işlemi yapmak istiyoruz ama burdaki Timestamp değişkeni object tipindedir. Bunu zaman değişkenine
# çevirmemiz gerekmektedir.
df.info()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Bununle beraber yapılan bütün yorumları gün cinsinden ifade etmemiz gerekmektedir.
# Bugünün tarihi diye bir tarih belirleyeceğiz ve bu tarihten yorumların yapıldığı tarihi çıkaracağız.

# En son yapılan tarihe bakıyoruz
df["Timestamp"].max()
# Son yapılan yorum tarihinden 5 gün sonrasında olduğumuzu varsayarak analizlerimize başlayacağız.
current_date = pd.to_datetime("2021-02-10")
# Yorumları gün cinsinden ifade ettiğimiz bir değişken ekliyoruz.
df["Days"] = (current_date - df["Timestamp"]).dt.days


# Farklı zaman aralıklarına, farklı bir şekilde odaklanacağız. Bunun için her birine farklı bir ağırlık vererek zamanın etkisini ağırlık hesabına yansıtacağız.
# Sırasıyla belirlediğimiz zaman aralıklarına (0-30-90-180-sonrası) %28, %26, %24 ve %22 ağırlıklarını verdik.
df.loc[df["Days"] <= 30, "Rating"].mean() * 28/100 + \
       df.loc[(df["Days"] > 30) & (df["Days"] <= 90), "Rating"].mean() * 26/100 + \
       df.loc[(df["Days"] > 90) & (df["Days"] <= 180), "Rating"].mean() * 24/100 + \
       df.loc[(df["Days"] > 180), "Rating"].mean() * 22/100

# Yorum -----> İşlem sonucunda 4.765 gibi bir değer çıktı. Fakat buradada şöyle bir problem olmaktadır; Kursun %100' ünü izleyen bir kullanıcı ile kursun %5' ini
# izleyen bir kullanıcının verdiği puanın ağırlıkları aynı olmamalıdır.



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uygulama 3 - User-Based Weighted Average (Kullanıcı Ağırlıklı Ortalama)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Aslında müşteri kalitesine göre ağırlıklandırmalar yapıyoruz.
df.groupby("Progress").agg({"Rating": "mean",
                            "Progress": "count"})

# Farklı izlenme durumlarına göre ağırlık hesabı yaptık. Kendimizin belirlediği aralıkların sırasıyla ağırlıkları %22, %24, %26 ve %28' dir.
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100


# ! DİKKAT ! = Yöntemler sonucunda farklı çıktılar alıyoruz. Hangisinin daha iyi olduğu hakkında tartışmıyoruz. İlgilendiğimiz şey ortalama işini
#              hassaslaştırdığımızda ortalamanın farklılaştığıdır.


# Yorum ------> Bundan önceki yöntemlerden farklı olarak muhattabımıza söyleyebileceğimiz şeyler var. Örneğin kursun beğenilme trendini son zamanlara göre hassaslaştırdık.
#               Bu yöntem tek başına kullanıldığında da izlenme tarihleri kıssası göz ardı edilmiş olur. Bu durumda yapılması gereken şey son 2 uygulamayı bir araya getirmektir.
#               İki durum için de istediğimiz ağırlığı verebiliriz. Bu olay sonucunda muhattabımıza karşı elmizde daha dolu bir argüman olur.



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uygulama 4 - Weighted Rating (Ağırlıklı Ortalama)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["Days"] <= 30, "Rating"].mean()* w1 / 100 + \
           dataframe.loc[(dataframe["Days"] > 30) & (dataframe["Days"] <= 90), "Rating"].mean()* w2 / 100  + \
           dataframe.loc[(dataframe["Days"] > 90) & (dataframe["Days"] <= 180), "Rating"].mean()* w3 /100 + \
           dataframe.loc[(dataframe["Days"] > 180), "Rating"].mean()* w4 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

# Yorum -----> Sonucumuz 4.7826 küsüratlı çıkmaktadır. Bu uygulama ilk birden fazla faktörü göz önünde bulundurmuş oluyoruz ve böylelikle daha sağlıklı bir analiz yapabiliyoruz.
#              Bu sebeple herhangi bir ürünü Rating puanına göre sıraladığımızda, sıralamamızı nelere göre yapmamız gerektiğini bilmiş olduk.
