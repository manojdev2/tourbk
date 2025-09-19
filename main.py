from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
import json
import re
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Generative AI Trip Planner with Google GenAI SDK and Weather")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
GOOGLE_MAPS_API_KEY = "AIzaSyDAUhNkL--7MVKHtlFuR3acwa7ED-cIoAU"
WEATHER_API_KEY = "6419738e339e4507aa8122732240910"
WEATHER_API_URL = "http://api.weatherapi.com/v1"

# --- Models ---
class TripRequest(BaseModel):
    location: str
    duration: int
    budget: int
    theme: str
    start_date: Optional[str] = None
    traveler_count: Optional[int] = 1
    preferred_transport: Optional[str] = "driving"
    from_location: Optional[str] = None
    to_location: Optional[str] = None

class Activity(BaseModel):
    name: str
    description: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    estimated_cost: Optional[int] = None
    duration_hours: Optional[float] = None
    category: Optional[str] = None
    best_time: Optional[str] = None  # e.g., "10:00 AM - 1:00 PM"

class WeatherForecast(BaseModel):
    date: str
    condition: str
    max_temp_c: float
    min_temp_c: float
    chance_of_rain: float

class ItineraryDay(BaseModel):
    day: int
    date: Optional[str] = None
    activities: List[Activity]
    weather: Optional[WeatherForecast] = None
    total_day_cost: Optional[int] = None

class Hotel(BaseModel):
    name: str
    address: str
    rating: Optional[float] = None
    price_level: Optional[int] = None
    latitude: float
    longitude: float
    place_id: str
    photo_reference: Optional[str] = None

class RouteDetails(BaseModel):
    distance: str
    duration: str
    travel_mode: str
    estimated_cost: Optional[int] = None
    polyline: Optional[str] = None
    steps: Optional[List[dict]] = None

class Itinerary(BaseModel):
    location: str
    duration: int
    budget: int
    theme: str
    start_date: Optional[str] = None
    traveler_count: Optional[int] = None
    preferred_transport: Optional[str] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    days: List[ItineraryDay]
    total_estimated_cost: Optional[int] = None
    hotels: Optional[List[Hotel]] = None
    route_details: Optional[RouteDetails] = None

class BookingRequest(BaseModel):
    itinerary_id: str
    user_id: str
    payment_token: str

class LocationCoordinates(BaseModel):
    location: str
    latitude: float
    longitude: float

# --- Booking storage (in-memory) ---
bookings = {}

# --- Google Maps API Functions ---
def get_location_coordinates_from_google(location: str) -> dict:
    url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': location, 'key': GOOGLE_MAPS_API_KEY}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            location_data = data['results'][0]['geometry']['location']
            return {
                'latitude': location_data['lat'],
                'longitude': location_data['lng'],
                'formatted_address': data['results'][0]['formatted_address']
            }
    except Exception as e:
        print(f"Error getting coordinates from Google: {e}")
    return get_location_coordinates_dict(location)

def find_hotels_near_location(location: str, radius: int = 5000) -> List[Hotel]:
    coords = get_location_coordinates_from_google(location)
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f"{coords['latitude']},{coords['longitude']}",
        'radius': radius,
        'type': 'lodging',
        'key': GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        hotels = []
        if data['status'] == 'OK':
            for place in data.get('results', [])[:10]:
                hotel = Hotel(
                    name=place.get('name', 'Unknown Hotel'),
                    address=place.get('vicinity', 'Address not available'),
                    rating=place.get('rating'),
                    price_level=place.get('price_level'),
                    latitude=place['geometry']['location']['lat'],
                    longitude=place['geometry']['location']['lng'],
                    place_id=place.get('place_id', ''),
                    photo_reference=place.get('photos', [{}])[0].get('photo_reference') if place.get('photos') else None
                )
                hotels.append(hotel)
        return hotels
    except Exception as e:
        print(f"Error finding hotels: {e}")
        return []

def get_route_details(from_location: str, to_location: str, travel_mode: str = "driving") -> Optional[RouteDetails]:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    mode_mapping = {
        "driving": "driving", "car": "driving", "walking": "walking",
        "transit": "transit", "public_transport": "transit",
        "bicycling": "bicycling", "bike": "bicycling", "motorcycle": "motorcycle", "flight": "flight"
    }
    google_mode = mode_mapping.get(travel_mode.lower(), "driving")
    params = {
        'origin': from_location,
        'destination': to_location,
        'mode': google_mode,
        'key': GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == 'OK' and data['routes']:
            route = data['routes'][0]
            leg = route['legs'][0]
            estimated_cost = calculate_travel_cost(leg['distance']['value'], google_mode)
            return RouteDetails(
                distance=leg['distance']['text'],
                duration=leg['duration']['text'],
                travel_mode=google_mode,
                estimated_cost=estimated_cost,
                polyline=route.get('overview_polyline', {}).get('points'),
                steps=[{
                    'instruction': step.get('html_instructions', ''),
                    'distance': step.get('distance', {}).get('text', ''),
                    'duration': step.get('duration', {}).get('text', ''),
                    'start_location': step.get('start_location', {}),
                    'end_location': step.get('end_location', {})
                } for step in leg.get('steps', [])[:5]]
            )
    except Exception as e:
        print(f"Error getting route details: {e}")
    return None

def calculate_travel_cost(distance_meters: int, travel_mode: str) -> int:
    distance_km = distance_meters / 1000
    cost_rates = {
        "driving": 8,  # ₹8 per km
        "transit": 2,  # ₹2 per km
        "bicycling": 5,  # ₹5 per km
        "walking": 0
    }
    return int(distance_km * cost_rates.get(travel_mode, 10))

# --- Weather API Functions ---
def get_weather_forecast(location: str, start_date: str, days: int) -> List[WeatherForecast]:
    """Get weather forecast for the trip duration"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()
        url = f"{WEATHER_API_URL}/forecast.json"
        forecasts = []
        for i in range(days):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            params = {
                'key': WEATHER_API_KEY,
                'q': location,
                'dt': date,
                'days': 1
            }
            response = requests.get(url, params=params)
            data = response.json()
            if 'forecast' in data and data['forecast']['forecastday']:
                forecast_day = data['forecast']['forecastday'][0]
                forecasts.append(WeatherForecast(
                    date=date,
                    condition=forecast_day['day']['condition']['text'],
                    max_temp_c=forecast_day['day']['maxtemp_c'],
                    min_temp_c=forecast_day['day']['mintemp_c'],
                    chance_of_rain=forecast_day['day']['daily_chance_of_rain']
                ))
        return forecasts
    except Exception as e:
        print(f"Error getting weather forecast: {e}")
        return []

# --- GenAI client ---
def get_genai_model():
    genai.configure(api_key='AIzaSyDYRX4tB69CrWFdfEer9uyzpNANuADCwqc')
    return genai.GenerativeModel("gemini-1.5-flash")

def clean_json_string(json_string: str) -> str:
    if "```json" in json_string:
        json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
    json_string = json_string.strip()
    json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)
    json_string = re.sub(r'(\w+)(\s*:)', r'"\1"\2', json_string)
    json_string = re.sub(r'""(\w+)""(\s*:)', r'"\1"\2', json_string)
    return json_string

def validate_json_structure(data: dict) -> bool:
    if not isinstance(data, dict) or "days" not in data or not isinstance(data["days"], list):
        return False
    for day in data["days"]:
        if not isinstance(day, dict) or "day" not in day or "activities" not in day or not isinstance(day["activities"], list):
            return False
        if len(day["activities"]) < 2 or len(day["activities"]) > 3:
            return False
        for activity in day["activities"]:
            if not isinstance(activity, dict) or "name" not in activity or "best_time" not in activity:
                return False
    return True

def generate_itinerary_with_genai(user_inputs: dict) -> dict:
    model = get_genai_model()
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    traveler_count = user_inputs.get('traveler_count', 1)
    budget = user_inputs['budget']
    duration = user_inputs['duration']
    theme = user_inputs['theme']
    start_date = user_inputs.get('start_date')
    preferred_transport = user_inputs.get('preferred_transport', 'any')

    traveler_info = f"for a group of {traveler_count} travelers" if traveler_count else ""
    start_date_info = f"starting from {start_date}" if start_date else ""
    transport_info = f"preferring {preferred_transport} transport" if preferred_transport else ""

    # Adjust budget per day and per activity
    budget_per_day = budget // duration
    budget_per_activity = budget_per_day // 3  # Max 3 activities

    prompt = (
        f"Generate a detailed {duration}-day travel itinerary for {destination} "
        f"focused on {theme} theme, {traveler_info}, with a total budget of INR {budget} "
        f"for all travelers and all days {start_date_info} {transport_info}. "
        f"Distribute activities and costs such that the sum total for all travelers and days does not exceed INR {budget}. "
        f"Each activity's estimated cost should reflect the total for the entire group, with a maximum of INR {budget_per_activity} per activity. "
        f"Return ONLY a valid JSON object with no additional text, comments, or markdown formatting. "
        f"The JSON structure should be:\n\n"
        "{\n"
        '  "days": [\n'
        "    {\n"
        '      "day": 1,\n'
        '      "activities": [\n'
        "        {\n"
        '          "name": "Exact place name (real attraction, landmark, museum, park, restaurant, etc.)",\n'
        '          "description": "Detailed description (2-3 sentences)",\n'
        '          "latitude": 12.9716,\n'
        '          "longitude": 77.5946,\n'
        '          "estimated_cost": 1000,\n'
        '          "duration_hours": 2.5,\n'
        '          "category": "sightseeing",\n'
        '          "best_time": "10:00 AM - 1:00 PM"\n'
        "        }\n"
        "      ],\n"
        '      "total_day_cost": 3000\n'
        "    }\n"
        "  ],\n"
        '  "total_estimated_cost": 12000\n'
        "}\n\n"
        "CRITICAL REQUIREMENTS:\n"
        f"- Create exactly {duration} days\n"
        "- Include at least 2 and up to 3 activities per day\n"
        "- All activity names MUST be real and specific places within or near the destination "
        "(e.g., Marina Beach, Kapaleeshwarar Temple, Santhome Basilica, Guindy National Park).\n"
        "- Do NOT use generic labels like 'Activity 1', 'Explore Adyar', or 'Local Market'.\n"
        "- Provide accurate latitude and longitude coordinates for each real place.\n"
        f"- Include realistic costs in INR within the total group budget (max INR {budget_per_activity} per activity).\n"
        f"- Focus on {theme} theme.\n"
        "- Use only these categories: sightseeing, food, adventure, cultural, shopping, nature, nightlife, heritage.\n"
        "- Ensure valid JSON format with no comments, no trailing commas, no extra text.\n"
        "- All property names and string values must be in double quotes.\n"
        f"- Keep total_estimated_cost under INR {budget} for the entire group.\n"
        "- Include duration_hours as decimal numbers (e.g., 2.5).\n"
        "- Include best_time for each activity in format 'HH:MM AM/PM - HH:MM AM/PM'.\n"
        f"- Consider the preferred transport mode: {preferred_transport}.\n"
        "- Suggest activities suitable for typical weather conditions in the destination.\n"
    )


    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "candidate_count": 1,
                },
            )
            output_text = response.text or ""
            cleaned_json = clean_json_string(output_text)
            print(f"Attempt {attempt + 1} - Cleaned JSON: {cleaned_json[:500]}...")
            try:
                parsed_data = json.loads(cleaned_json)
                if validate_json_structure(parsed_data):
                    if parsed_data.get('total_estimated_cost', 0) <= budget:
                        print(f"Successfully parsed JSON and validated budget on attempt {attempt + 1}")
                        return parsed_data
                    else:
                        print(f"Total estimated cost exceeds budget on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            return create_fallback_itinerary(user_inputs)
                else:
                    print(f"Invalid JSON structure on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return create_fallback_itinerary(user_inputs)
            except json.JSONDecodeError as json_error:
                print(f"JSON Parse Error (attempt {attempt + 1}): {json_error}")
                if attempt == max_retries - 1:
                    return create_fallback_itinerary(user_inputs)
        except Exception as e:
            print(f"Error calling GenAI SDK (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return create_fallback_itinerary(user_inputs)
    return {}

def create_fallback_itinerary(user_inputs: dict) -> dict:
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    location_coords = get_location_coordinates_dict(destination)
    traveler_count = user_inputs.get('traveler_count', 1)
    budget_per_day = user_inputs['budget'] // user_inputs['duration']
    cost_per_activity = budget_per_day // 3
    fallback_data = {
        "days": [],
        "total_estimated_cost": user_inputs['budget']
    }
    for day_num in range(1, user_inputs['duration'] + 1):
        day_activities = []
        for i in range(2):  # 2 activities per day
            activity = {
                "name": f"Explore {destination} - Activity {i + 1}",
                "description": f"Discover the {user_inputs['theme']} aspects of {destination}.",
                "latitude": location_coords['latitude'] + (i * 0.01),
                "longitude": location_coords['longitude'] + (i * 0.01),
                "estimated_cost": cost_per_activity,
                "duration_hours": 2.5,
                "category": user_inputs['theme'] if user_inputs['theme'] in ['sightseeing', 'food', 'adventure', 'cultural', 'shopping', 'nature', 'nightlife', 'heritage'] else 'sightseeing',
                "best_time": f"{10 + i*3}:00 AM - {13 + i*3}:00 PM"
            }
            day_activities.append(activity)
        fallback_data["days"].append({
            "day": day_num,
            "activities": day_activities,
            "total_day_cost": cost_per_activity * len(day_activities)
        })
    return fallback_data

def get_location_coordinates_dict(location: str) -> dict:
    location_coords = {
        "bangalore": {"latitude": 12.9716, "longitude": 77.5946},
        "mumbai": {"latitude": 19.0760, "longitude": 72.8777},
        "delhi": {"latitude": 28.7041, "longitude": 77.1025},
        "goa": {"latitude": 15.2993, "longitude": 74.1240},
        "kerala": {"latitude": 10.8505, "longitude": 76.2711},
        "rajasthan": {"latitude": 27.0238, "longitude": 74.2179},
        "jaipur": {"latitude": 26.9124, "longitude": 75.7873},
        "chennai": {"latitude": 13.0827, "longitude": 80.2707},
        "hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
        "pune": {"latitude": 18.5204, "longitude": 73.8567},
        "kolkata": {"latitude": 22.5726, "longitude": 88.3639},
        "melmaruvathur": {"latitude": 12.4801, "longitude": 79.8547},
        "paris": {"latitude": 48.8566, "longitude": 2.3522},
        "london": {"latitude": 51.5074, "longitude": -0.1278},
        "tokyo": {"latitude": 35.6762, "longitude": 139.6503},
        "new york": {"latitude": 40.7128, "longitude": -74.0060}
    }
    location_lower = location.lower()
    return location_coords.get(location_lower, {"latitude": 12.9716, "longitude": 77.5946})

def validate_coordinates(latitude: float, longitude: float) -> bool:
    return -90 <= latitude <= 90 and -180 <= longitude <= 180

def calculate_total_cost(days: List[ItineraryDay]) -> int:
    total = 0
    for day in days:
        day_total = day.total_day_cost or sum(activity.estimated_cost or 0 for activity in day.activities)
        total += day_total
    return total

def process_payment(payment_token: str) -> bool:
    return True

# --- API Routes ---
@app.post("/trip/generate-itinerary", response_model=Itinerary)
async def generate_itinerary(req: TripRequest):
    itinerary_data = generate_itinerary_with_genai(req.dict())
    if not itinerary_data or "days" not in itinerary_data:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary")

    hotels = find_hotels_near_location(req.to_location or req.location) if req.to_location or req.location else []
    route_details = get_route_details(req.from_location, req.to_location, req.preferred_transport or "driving") if req.from_location and req.to_location else None
    weather_forecasts = get_weather_forecast(req.to_location or req.location, req.start_date, req.duration) if req.start_date else []

    try:
        days = []
        start_date = datetime.strptime(req.start_date, "%Y-%m-%d") if req.start_date else datetime.now()
        for idx, day_data in enumerate(itinerary_data.get("days", [])):
            activities = []
            for activity_data in day_data.get("activities", []):
                lat = activity_data.get("latitude")
                lng = activity_data.get("longitude")
                if lat is not None and lng is not None:
                    try:
                        lat = float(lat)
                        lng = float(lng)
                        if not validate_coordinates(lat, lng):
                            lat, lng = None, None
                    except (ValueError, TypeError):
                        lat, lng = None, None
                cost = int(activity_data.get("estimated_cost", 0) * req.traveler_count) if activity_data.get("estimated_cost") else 0
                duration = float(activity_data.get("duration_hours")) if activity_data.get("duration_hours") else None
                activity = Activity(
                    name=activity_data.get("name", "Unknown Activity"),
                    description=activity_data.get("description", "No description available"),
                    latitude=lat,
                    longitude=lng,
                    estimated_cost=cost,
                    duration_hours=duration,
                    category=activity_data.get("category", "general"),
                    best_time=activity_data.get("best_time", "9:00 AM - 12:00 PM")
                )
                activities.append(activity)
            day_cost = int(day_data.get("total_day_cost", 0) * req.traveler_count) if day_data.get("total_day_cost") else None
            day = ItineraryDay(
                day=day_data.get("day", 0),
                date=(start_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
                activities=activities,
                weather=weather_forecasts[idx] if idx < len(weather_forecasts) else None,
                total_day_cost=day_cost
            )
            days.append(day)
        total_cost = int(itinerary_data.get("total_estimated_cost", calculate_total_cost(days)))
        if route_details and route_details.estimated_cost:
            total_cost += route_details.estimated_cost
        return Itinerary(
            location=req.location,
            duration=req.duration,
            budget=req.budget,
            theme=req.theme,
            start_date=req.start_date,
            traveler_count=req.traveler_count,
            preferred_transport=req.preferred_transport,
            from_location=req.from_location,
            to_location=req.to_location,
            days=days,
            total_estimated_cost=total_cost,
            hotels=hotels,
            route_details=route_details
        )
    except Exception as e:
        print(f"Error parsing itinerary data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse itinerary data: {str(e)}")

@app.post("/trip/book-itinerary")
async def book_itinerary(req: BookingRequest):
    if not process_payment(req.payment_token):
        raise HTTPException(status_code=400, detail="Payment failed")
    bookings[req.itinerary_id] = {
        "user_id": req.user_id,
        "status": "booked",
        "booking_timestamp": datetime.now().isoformat()
    }
    return {
        "status": "booked",
        "itinerary_id": req.itinerary_id,
        "user_id": req.user_id,
        "message": "Itinerary successfully booked!"
    }

@app.get("/trip/bookings/{user_id}")
async def get_user_bookings(user_id: str):
    user_bookings = {
        booking_id: booking_data
        for booking_id, booking_data in bookings.items()
        if booking_data["user_id"] == user_id
    }
    return {
        "user_id": user_id,
        "bookings": user_bookings,
        "total_bookings": len(user_bookings)
    }

@app.get("/trip/coordinates/{location}")
async def get_location_coordinates(location: str):
    coords_dict = get_location_coordinates_from_google(location)
    return LocationCoordinates(
        location=location,
        latitude=coords_dict["latitude"],
        longitude=coords_dict["longitude"]
    )

@app.get("/trip/hotels/{location}")
async def get_hotels(location: str, radius: int = 5000):
    hotels = find_hotels_near_location(location, radius)
    return {
        "location": location,
        "hotels": hotels,
        "count": len(hotels)
    }

@app.get("/trip/route")
async def get_route(from_location: str, to_location: str, travel_mode: str = "driving"):
    route = get_route_details(from_location, to_location, travel_mode)
    if route:
        return route
    else:
        raise HTTPException(status_code=404, detail="Route not found")

@app.get("/status")
async def status():
    return {
        "status": "Generative AI Trip Planner with Google GenAI, Maps, and Weather API is running",
        "bookings_count": len(bookings),
        "features": [
            "AI-powered itinerary generation",
            "2-3 activities per day with best timing",
            "Weather forecast integration",
            "Activity coordinates for mapping",
            "Cost estimation with traveler count support",
            "Hotel recommendations via Google Places API",
            "Route planning via Google Directions API",
            "Multi-transport mode support",
            "Booking system",
            "Theme-based recommendations",
            "Robust JSON parsing",
            "Fallback itinerary generation"
        ],
        "supported_transport_modes": [
            "driving", "car", "walking", "transit",
            "public_transport", "bicycling", "bike"
        ],
        "version": "3.1"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Enhanced AI Trip Planner API",
        "description": "Generate personalized travel itineraries with AI, hotel recommendations, route planning, and weather forecasts",
        "endpoints": {
            "generate_itinerary": "/trip/generate-itinerary",
            "book_itinerary": "/trip/book-itinerary",
            "get_coordinates": "/trip/coordinates/{location}",
            "get_hotels": "/trip/hotels/{location}",
            "get_route": "/trip/route?from_location=X&to_location=Y&travel_mode=Z",
            "user_bookings": "/trip/bookings/{user_id}",
            "status": "/status"
        },
        "new_features": [
            "2-3 activities per day with best timing",
            "Weather forecast integration",
            "Hotel recommendations from Google Places",
            "Route planning with multiple transport modes",
            "Multi-traveler cost calculation",
            "Enhanced location geocoding"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
