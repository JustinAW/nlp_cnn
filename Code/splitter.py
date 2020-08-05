import csv

#***********************************************#
#       Hate Speech Dataset Splitter            #
#   Authors: Justin Weigle                      #
#   Edited: 17 Sep 2019                         #
#***********************************************#

def split(hs_data):
    hs = []
    ofn = []
    nei = []
    # remove the header line
    hs_data.pop(0)
    # sort by class
    for row in hs_data:
        if(row[0] == "0"):
            hs.append(row[1])
        if(row[0] == "1"):
            ofn.append(row[1])
        if(row[0] == "2"):
            nei.append(row[1])

    return hs, ofn, nei

def clean(hs, ofn, nei):
    hs = [str(line).replace('"', '') for line in hs]
    hs = [str(line).replace('\n', ' ') for line in hs]
    hs = [str(line).replace(',', '') for line in hs]
    ofn = [str(line).replace('"', '') for line in ofn]
    ofn = [str(line).replace('\n', ' ') for line in ofn]
    ofn = [str(line).replace(',', '') for line in ofn]
    nei = [str(line).replace('"', '') for line in nei]
    nei = [str(line).replace('\n', ' ') for line in nei]
    nei = [str(line).replace(',', '') for line in nei]
    
    return hs, ofn, nei

if __name__=="__main__":
    f = open("../Datasets/HATE_SPEECH/labeled_data.csv", 'r')
    reader = csv.reader(f)
    hs_data = []
    for row in reader:
        hs_data.append(row)
    f.close()

    print("Splitting the data into classes...")
    hs, ofn, nei = split(hs_data)
    print("Done!")
    print("Number of tweets classified as hate speech: " + str(len(hs)))
    print("Number of tweets classified as offensive: " + str(len(ofn)))
    print("Number of tweets classified as neither: " + str(len(nei)))
    total_tweets = len(hs) + len(ofn) + len(nei)
    print("Total tweets = " + str(total_tweets))
    print("Cleaning data...")
    hs, ofn, nei = clean(hs, ofn, nei)
    print("Cleaned!")
    print("Creating separate files for each class...")
    print("Creating hate speech file...")
    f = open("../Datasets/HATE_SPEECH/hs.csv", 'w')
    writer = csv.writer(f, delimiter = '\n')
    writer.writerow(hs)
    f.close()
    print("Creating offensive file...")
    f = open("../Datasets/HATE_SPEECH/ofn.csv", 'w', newline = '')
    writer = csv.writer(f, delimiter = '\n')
    writer.writerow(ofn)
    f.close()
    print("Creating neither file...")
    f = open("../Datasets/HATE_SPEECH/nei.csv", 'w', newline = '')
    writer = csv.writer(f, delimiter = '\n')
    writer.writerow(nei)
    f.close()
