from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback

# --- Main Setup ---
app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("bike_price_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Centralized Data Mappings ---
DATA_MAPPINGS = {
    "brands": {
        'aprilia': 0, 'ather': 1, 'bajaj': 2, 'benelli': 3, 'bmw': 4, 
        'cf moto': 5, 'ducati': 6, 'harley-davidson': 7, 'hero': 8, 'honda': 9, 
        'husqvarna': 10, 'indian': 11, 'java': 12, 'kawasaki': 13, 'ktm': 14, 
        'moto guzzi': 15, 'mv': 16, 'royal enfield': 17, 'suzuki': 18, 
        'triumph': 19, 'tvs': 20, 'vespa': 21, 'yamaha': 22
    },
    "models": {
        'aprilia sr 150': 0, 'ather energy ather 450x': 1, 'bajaj avenger cruise 220': 2, 
        'bajaj avenger street 160': 3, 'bajaj chetak': 4, 'bajaj dominar 250': 5, 
        'bajaj dominar 400': 6, 'bajaj pulsar 150': 7, 'bajaj pulsar ns160': 8, 
        'bajaj pulsar ns200': 9, 'bajaj pulsar rs200': 10, 'benelli imperiale 400': 11, 
        'benelli trk 502': 12, 'bmw f750 gs 850cc': 13, 'bmw g 310 gs': 14, 
        'bmw g 310 r': 15, 'bmw s 1000 rr': 16, 'bmw s 1000 xr': 17, 'cf moto 650nk': 18, 
        'ducati multistrada 1200': 19, 'ducati panigale v4': 20, 'ducati scrambler 800': 21, 
        'harley-davidson fat boy 1868cc': 22, 'harley-davidson forty-eight 1200cc': 23, 
        'harley-davidson iron 883 883cc': 24, 'harley-davidson street 750': 25, 
        'harley-davidson street rod 750cc': 26, 'hero glamour 125': 27, 
        'hero hf deluxe 100cc': 28, 'hero maestro edge 125': 29, 
        'hero passion pro 110cc': 30, 'hero pleasure plus 110cc': 31, 
        'hero splendor ismart 110cc': 32, 'hero splendor plus 100cc': 33, 
        'hero xpulse 200': 34, 'hero xtreme 160r': 35, 'honda activa 125': 36, 
        'honda activa 5g': 37, 'honda activa 6g': 38, 'honda cb hornet 160r': 39, 
        'honda cb shine': 40, 'honda cbr 650r': 41, 'honda dio': 42, 
        'honda gold wing 1800cc': 43, 'honda livo 110cc': 44, 
        'husqvarna svartpilen 250': 45, 'husqvarna vitpilen 250': 46, 
        'indian chief classic 1800cc': 47, 'java 42': 48, 'java perak': 49, 
        'kawasaki ninja 1000': 50, 'kawasaki ninja 300': 51, 'kawasaki ninja 650': 52, 
        'kawasaki versys 1000': 53, 'kawasaki versys 650': 54, 'kawasaki z650': 55, 
        'kawasaki z900': 56, 'ktm 125 duke': 57, 'ktm 200 duke': 58, 'ktm 250 duke': 59, 
        'ktm 390 duke': 60, 'ktm rc 125': 61, 'ktm rc 200': 62, 'ktm rc 390': 63, 
        'moto guzzi v85 tt 850cc': 64, 'mv agusta f3 800': 65, 
        'royal enfield bullet 350': 66, 'royal enfield classic 350': 67, 
        'royal enfield classic 500': 68, 'royal enfield continental gt 650': 69, 
        'royal enfield himalayan': 70, 'royal enfield interceptor 650': 71, 
        'royal enfield meteor 350': 72, 'royal enfield thunderbird 350': 73, 
        'suzuki access 125': 74, 'suzuki burgman street 125': 75, 
        'suzuki gixxer sf 150': 76, 'suzuki hayabusa 1340cc': 77, 'suzuki intruder 150': 78, 
        'triumph bonneville t100': 79, 'triumph bonneville t120': 80, 
        'triumph rocket 3 2500cc': 81, 'triumph street triple 765': 82, 
        'triumph trident 660': 83, 'tvs apache rtr 160': 84, 'tvs apache rtr 160 4v': 85, 
        'tvs apache rtr 200 4v': 86, 'tvs jupiter 110': 87, 'tvs ntorq 125': 88, 
        'tvs rr310': 89, 'tvs star city plus 110cc': 90, 'vespa sxl 125': 91, 
        'yamaha fz s fi': 92, 'yamaha mt-15': 93, 'yamaha yzf-r15 v3 150cc': 94
    },
    # --- THIS IS THE CRITICAL FIX ---
    # The full and correct list of locations your model was trained on.
    "locations": {
        '24 parganas':0, 'adhartal':1, 'adoni':2, 'adyar':3, 'agartala':4, 'agra':5, 'ahmedabad':6, 'ahmednagar':7, 
        'aizawl':8, 'ajmer':9, 'akola':10, 'alandi':11, 'alappuzha':12, 'aligarh':13, 'alipurduar':14, 'allahabad':15, 
        'alwar':16, 'ambala':17, 'ambedkarnagar':18, 'ambikapur':19, 'amravati':20, 'amreli':21, 'amritsar':22, 
        'anand':23, 'anantapur':24, 'angul':25, 'ankleshwar':26, 'ankola':27, 'arambol':28, 'araria':29, 'arasikere':30, 
        'ariyalur':31, 'arrah':32, 'asansol':33, 'aurangabad':34, 'azamgarh':35, 'baddi':36, 'badlapur':37, 'bagalkot':38, 
        'bageshwar':39, 'baghpat':40, 'bahadurgarh':41, 'bahraich':42, 'baksa':43, 'balaghat':44, 'balangir':45, 
        'balasore':46, 'ballia':47, 'bally':48, 'balotra':49, 'balrampur':50, 'balurghat':51, 'banaskantha':52, 'banda':53, 
        'bandipora':54, 'bangalore':55, 'banka':56, 'bankura':57, 'banswara':58, 'bapatla':59, 'barabanki':60, 'baramati':61, 
        'baran':62, 'bardhaman':63, 'bardoli':64, 'bareilly':65, 'bargarh':66, 'barwani':67, 'basirhat':68, 'basti':69, 
        'bathinda':70, 'beed':71, 'begusarai':72, 'belgaum':73, 'bellary':74, 'berhampore':75, 'berhampur':76, 'bettiah':77, 
        'bhadrak':78, 'bhagalpur':79, 'bhandara':80, 'bharuch':81, 'bhatkal':82, 'bhavnagar':83, 'bhilai':84, 'bhilwara':85, 
        'bhimavaram':86, 'bhiwadi':87, 'bhiwani':88, 'bhongir':89, 'bhopal':90, 'bhubaneswar':91, 'bhuj':92, 'bidar':93, 
        'bijapur':94, 'bijnor':95, 'bikaner':96, 'bilaspur':97, 'birbhum':98, 'bishnupur':99, 'bongaigaon':100, 'bordi':101, 
        'botad':102, 'boudh':103, 'budaun':104, 'bulandshahr':105, 'buldhana':106, 'bundi':107, 'bongaon':108, 'calicut':109, 
        'chamarajanagar':110, 'chamba':111, 'chamoli':112, 'champawat':113, 'champhai':114, 'chandauli':115, 'chandel':116, 
        'chandigarh':117, 'chandrapur':118, 'changlang':119, 'charkhi dadri':120, 'chennai':121, 'chhapra':122, 
        'chhatarpur':123, 'chhindwara':124, 'chikmagalur':125, 'chirala':126, 'chirang':127, 'chitradurga':128, 
        'chitrakoot':129, 'chittoor':130, 'chittorgarh':131, 'churachandpur':132, 'churu':133, 'coimbatore':134, 'cuddalore':135, 
        'cuttack':136, 'cooch behar':137, 'dahod':138, 'dakshina kannada':139, 'darbhanga':140, 'darjeeling':141, 'darrang':142, 
        'datia':143, 'dausa':144, 'davangere':145, 'dehradun':146, 'delhi':147, 'deoghar':148, 'deoria':149, 'dewas':150, 
        'dhalai':151, 'dhamtari':152, 'dhanbad':153, 'dhar':154, 'dharmapuri':155, 'dharwad':156, 'dhemaji':157, 'dhenkanal':158, 
        'dholpur':159, 'dhubri':160, 'dhule':161, 'dibrugarh':162, 'dimapur':163, 'dindigul':164, 'dindori':165, 'dispur':166, 
        'durg':167, 'durgapur':168, 'east garo hills':169, 'east siang':170, 'ernakulam':171, 'erode':172, 'etah':173, 'etawah':174, 
        'faizabad':175, 'faridabad':176, 'faridkot':177, 'farrukhabad':178, 'fatehabad':179, 'fatehgarh sahib':180, 'fatehpur':181, 
        'fazilka':182, 'firozabad':183, 'firozpur':184, 'gadag':185, 'gadchiroli':186, 'gajapati':187, 'ganderbal':188, 
        'gandhidham':189, 'gandhinagar':190, 'ganganagar':191, 'gangtok':192, 'gautam budh nagar':193, 'gaya':194, 
        'ghaziabad':195, 'ghazipur':196, 'giridih':197, 'goalpara':198, 'gokak':199, 'golaghat':200, 'gonda':201, 
        'gondia':202, 'gopalganj':203, 'gorakhpur':204, 'guna':205, 'guntakal':206, 'guntur':207, 'gurdaspur':208, 
        'gurgaon':209, 'gwalior':210, 'hailakandi':211, 'hamirpur':212, 'hansi':213, 'hanumangarh':214, 'hapur':215, 'harda':216, 
        'hardoi':217, 'haridwar':218, 'hassan':219, 'hathras':220, 'haveri':221, 'hazaribagh':222, 'himatnagar':223, 
        'hisar':224, 'hojai':225, 'hooghly':226, 'hoshangabad':227, 'hoshiarpur':228, 'hospet':229, 'hosur':230, 'howrah':231, 
        'hubli':232, 'hyderabad':233, 'ichalkaranji':234, 'idukki':235, 'imphal':236, 'indore':237, 'itanagar':238, 'jabalpur':239, 
        'jagatsinghpur':240, 'jagtial':241, 'jajpur':242, 'jalandhar':243, 'jalaun':244, 'jalgaon':245, 'jalna':246, 'jalore':247, 
        'jalpaiguri':248, 'jammu':249, 'jamnagar':250, 'jamshedpur':251, 'jangaon':252, 'jaunpur':253, 'itanagar':254, 'jaipur':255, 
        'jhalawar':256, 'jhansi':257, 'jharsuguda':258, 'jhunjhunu':259, 'jind':260, 'jodhpur':261, 'jorhat':262, 'junagadh':263, 
        'kadi':264, 'kaithal':265, 'kakinada':266, 'kalahandi':267, 'kalaburagi':268, 'kamrup':269, 'kanchipuram':270, 
        'kanker':271, 'kannur':272, 'kanpur':273, 'kanyakumari':274, 'kapurthala':275, 'karaikal':276, 'karaikudi':277, 
        'karbi anglong':278, 'karimganj':279, 'karimnagar':280, 'karnal':281, 'karur':282, 'kasaragod':283, 'kathua':284, 
        'katihar':285, 'katni':286, 'kavali':287, 'khammam':288, 'khandwa':289, 'khanna':290, 'kheda':291, 'khordha':292, 
        'kolkata':293, 'kollam':294, 'koppal':295, 'korba':296, 'kota':297, 'kottayam':298, 'kozhikode':299, 'krishna':300, 
        'kulgam':301, 'kullu':302, 'kurnool':303, 'kurukshetra':304, 'lakhimpur':305, 'latur':306, 'lawngtlai':307, 'leh':308, 
        'lucknow':309, 'ludhiana':310, 'lunglei':311, 'machilipatnam':312, 'madhepura':313, 'madhubani':314, 'madurai':315, 
        'mahesana':316, 'malappuram':317, 'malda':318, 'malkangiri':319, 'mamit':320, 'mandi':321, 'mandya':322, 'mandsaur':323, 
        'mangalore':324, 'manipal':325, 'mansa':326, 'mathura':327, 'maunath bhanjan':328, 'mayurbhanj':329, 'medak':330, 
        'meerut':331, 'midnapore':332, 'mirzapur':333, 'moga':334, 'mohali':335, 'moradabad':336, 'morbi':337, 'morena':338, 
        'motihari':339, 'mumbai':340, 'munger':341, 'murshidabad':342, 'muzaffarnagar':343, 'muzaffarpur':344, 'mysore':345, 
        'nadiad':346, 'nagaon':347, 'nagapattinam':348, 'nagaur':349, 'nagpur':350, 'nalbari':351, 'nalgonda':352, 'namakkal':353, 
        'nanded':354, 'nandurbar':355, 'nandyal':356, 'narasaraopet':357, 'narnaul':358, 'narsinghpur':359, 'nashik':360, 
        'navi mumbai':361, 'navsari':362, 'nawabganj':363, 'nawada':364, 'nawanshahr':365, 'neemuch':366, 'nellore':367, 
        'new delhi':368, 'nizamabad':369, 'noida':370, 'north 24 parganas':371, 'ongole':372, 'ooty':373, 'palakkad':374, 
        'palani':375, 'palanpur':376, 'palwal':377, 'panchkula':378, 'panipat':379, 'panna':380, 'patiala':381, 'patna':382, 
        'pondicherry':383, 'pune':384, 'puri':385, 'purnia':386, 'raichur':387, 'raipur':388, 'rajahmundry':389, 'rajkot':390, 
        'rajnandgaon':391, 'rajsamand':392, 'ramanathapuram':393, 'ramgarh':394, 'ranchi':395, 'ratlam':396, 'ratnagiri':397, 
        'rayagada':398, 'rewari':399, 'rohtak':400, 'ropar':401, 'rourkela':402, 'sagar':403, 'saharanpur':404, 'saharsa':405, 
        'salem':406, 'samastipur':407, 'sambalpur':408, 'sangli':409, 'sangrur':410, 'satara':411, 'satna':412, 'shimla':413, 
        'shimoga':414, 'shivamogga':415, 'sikar':416, 'silchar':417, 'siliguri':418, 'sirsa':419, 'sitapur':420, 'sivakasi':421, 
        'sivasagar':422, 'siwan':423, 'solan':424, 'solapur':425, 'sonipat':426, 'srinagar':427, 'srikakulam':428, 'surat':429, 
        'surendranagar':430, 'suryapet':431, 'tadepalligudem':432, 'tenali':433, 'thane':434, 'thanjavur':435, 'thrissur':436, 
        'tiruchirappalli':437, 'tirunelveli':438, 'tirupati':439, 'tiruppur':440, 'tiruvannamalai':441, 'tumkur':442, 
        'tuticorin':443, 'udaipur':444, 'udupi':445, 'ujjain':446, 'una':447, 'vadodara':448, 'valsad':449, 'vapi':450, 
        'varanasi':451, 'vasai':452, 'vellore':453, 'vidisha':454, 'vijayawada':455, 'visakhapatnam':456, 'vizianagaram':457, 
        'warangal':458, 'wardha':459, 'yamunanagar':460, 'yavatmal':461, 'zirakpur':462
    }
}


# Calculate the mean of all known location codes for the "Other" option
mean_location_code = np.mean(list(DATA_MAPPINGS['locations'].values()))
print(f"Calculated mean location code for 'Other': {mean_location_code}")


# --- API Endpoints ---
@app.route('/')
def home():
    return "✅ Flask API for Bike Price Prediction is running!"

@app.route('/get_data_mappings')
def get_data_mappings():
    """Provides all mappings to the frontend."""
    # To keep the frontend dropdown manageable, we'll only send a subset of locations
    # But the backend will use the full list for the "Other" calculation
    
    # Define a smaller list of major cities for the UI dropdown
    major_cities = {
        "bangalore": 55, "chennai": 121, "delhi": 147, "gurgaon": 209,
        "hyderabad": 233, "kolkata": 293, "mumbai": 340, "noida": 370,
        "pune": 384, "ahmedabad": 6
    }

    frontend_mappings = {
        "brands": DATA_MAPPINGS["brands"],
        "models": DATA_MAPPINGS["models"],
        "locations": major_cities # Send only the major cities to the frontend
    }
    return jsonify(frontend_mappings)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly. Check server logs.'}), 500
        
    try:
        data = request.get_json()
        
        location_code = data['location']
        if int(location_code) == -1:  # Handle the "Other" case
            location_feature_value = mean_location_code
        else:
            location_feature_value = location_code
        
        features = np.array([[
            data['brand'], 
            data['model'], 
            data['year'], 
            data['kilometers'],
            location_feature_value,
            data['power'],
            data['owner_Fourth Owner Or More'], 
            data['owner_Second Owner'],
            data['owner_Third Owner'],
            data['owner_Unknown']
        ]])

        prediction = model.predict(features)
        
        return jsonify({'prediction': float(prediction[0])})

    except KeyError as e:
        return jsonify({'error': f'Missing key in request: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction.', 'details': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
