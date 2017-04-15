# this scripy is based on python2 and don't support python3
import urllib2

response = urllib2.urlopen('http://www.math.pku.edu.cn')
html = response.read()
print html