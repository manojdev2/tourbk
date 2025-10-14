from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uvicorn
import json
import re
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Trip Planner with User Comments Support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KeysAIzaSyCP7SXQh5kgSk7wHiz2rgyqgs-0knykZxQ"
GOOGLE_MAPS_API_KEY = "
WEATHER_API_KEY = "6419738e339e4507aa8122732240910"
WEATHER_API_URL = "http://api.weatherapi.com/v1"

# --- Models ---
class TripRequest(BaseModel):
    location: str
    duration: int
    budget: int
    themes: List[str]  # Changed from theme: str to themes: List[str]
    start_date: Optional[str] = None
    traveler_count: Optional[int] = 1
    preferred_transport: Optional[str] = "driving"
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    user_comments: Optional[str] = None

class Activity(BaseModel):
    name: str
    description: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    estimated_cost: Optional[int] = None
    duration_hours: Optional[float] = None
    category: Optional[str] = None
    best_time: Optional[str] = None

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
    price_per_night: int
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
    themes: List[str]
    start_date: Optional[str] = None
    traveler_count: Optional[int] = None
    preferred_transport: Optional[str] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    user_comments: Optional[str] = None
    days: List[ItineraryDay]
    total_estimated_cost: Optional[int] = None
    hotels: Optional[List[Hotel]] = None
    route_details: Optional[RouteDetails] = None
    cost_breakdown: Optional[Dict[str, int]] = None

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

# def find_hotels_near_location(location: str, radius: int = 5000) -> List[Hotel]:
#     coords = get_location_coordinates_from_google(location)
#     url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
#     params = {
#         'location': f"{coords['latitude']},{coords['longitude']}",
#         'radius': radius,
#         'type': 'lodging',
#         'key': GOOGLE_MAPS_API_KEY
#     }
#     try:
#         response = requests.get(url, params=params)
#         data = response.json()
#         hotels = []
#         if data['status'] == 'OK':
#             for place in data.get('results', [])[:10]:
#                 hotel = Hotel(
#                     name=place.get('name', 'Unknown Hotel'),
#                     address=place.get('vicinity', 'Address not available'),
#                     rating=place.get('rating'),
#                     price_level=place.get('price_level'),
#                     latitude=place['geometry']['location']['lat'],
#                     longitude=place['geometry']['location']['lng'],
#                     place_id=place.get('place_id', ''),
#                     photo_reference=place.get('photos', [{}])[0].get('photo_reference') if place.get('photos') else None
#                 )
#                 hotels.append(hotel)
#         return hotels
#     except Exception as e:
#         print(f"Error finding hotels: {e}")
#         return []

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
    genai.configure(api_key='AIzaSyAmTnj87jegLwFyWCM50vpq5D9zkSh0VBg')
    return genai.GenerativeModel("gemini-2.0-flash")

def clean_json_string(json_string: str) -> str:
    # Remove code fences
    if json_string.strip().startswith("```"):
        json_string = re.sub(r"```[a-zA-Z]*\n?", "", json_string)
        json_string = json_string.replace("```", "")
    
    # Strip leading/trailing spaces
    json_string = json_string.strip()
    
    # Remove single-line and block comments
    json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
    
    # Remove trailing commas before } or ]
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
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

# def generate_itinerary_with_genai(user_inputs: dict) -> dict:
#     model = get_genai_model()
#     destination = user_inputs.get('to_location') or user_inputs.get('location')
#     traveler_count = user_inputs.get('traveler_count', 1)
#     budget = user_inputs['budget']
#     duration = user_inputs['duration']
#     themes = user_inputs['themes']  # Now it's a list
#     start_date = user_inputs.get('start_date')
#     preferred_transport = user_inputs.get('preferred_transport', 'any')
#     user_comments = user_inputs.get('user_comments', '')

#     # Create themes string for prompt
#     themes_str = ", ".join(themes) if themes else "general"
#     primary_theme = themes[0] if themes else "cultural"

#     comments_info = ""
#     if user_comments:
#         comments_info = f"\n\nUSER PREFERENCES: \"{user_comments}\"\n" \
#                         f"Incorporate these preferences when selecting specific real places and activities."

#     budget_per_person = budget // traveler_count
#     budget_per_person_per_day = budget_per_person // duration
#     budget_per_activity_per_person = budget_per_person_per_day // 3

#     prompt = f"""You are a travel expert for {destination}. Generate a {duration}-day itinerary with REAL, SPECIFIC places only.

# DESTINATION: {destination}
# THEMES: {themes_str} (focus on only selected themes throughout the trip)
# BUDGET: INR {budget} total, for {traveler_count} traveler(s)
# ALL ACTIVITY COSTS ARE PER TRAVELER, NOT TOTAL.
# DATES: {start_date if start_date else "Flexible"}
# TRANSPORT: {preferred_transport}
# {comments_info}

# IMPORTANT: Mix activities from only selected themes ({themes_str}) across different days. Don't focus on just one theme per day.

# RETURN ONLY VALID JSON - NO other text, explanations, or markdown:

# {{
#   "days": [
#     {{
#       "day": 1,
#       "activities": [
#         {{
#           "name": "REAL PLACE NAME",
#           "description": "...",
#           "latitude": 00.0000,
#           "longitude": 00.0000,
#           "estimated_cost": {budget_per_activity_per_person},
#           "duration_hours": 2.5,
#           "category": "one_of_selected_themes",
#           "best_time": "10:00 AM - 1:00 PM"
#         }}
#       ],
#       "total_day_cost": {budget_per_person_per_day}
#     }}
#   ],
#   "total_estimated_cost": {budget_per_person}
# }}
# CRITICAL RULES:
# 1. Use ONLY real, famous, specific places in {destination}
# 2. NO generic names like "Activity 1" or "Local Market"
# 3. Each activity cost is per person (multiply by traveler_count for day and trip totals)
# 4. Per traveler daily cost must not exceed INR {budget_per_person_per_day}
# 5. All days combined for all travelers must not exceed total budget INR {budget}
# 6. Include activities from only selected themes: {themes_str}
# 7. Categories should match selected themes: {themes_str}
# 8. Include atleast 4-6 activities per day, if budget allows
# 9. Vary the theme categories across days for diversity
# """

#     try:
#         response = model.generate_content(
#             prompt,
#             generation_config={
#                 "temperature": 0.3,
#                 "top_p": 0.7,
#                 "candidate_count": 1,
#             },
#         )

#         output_text = response.text or ""
#         cleaned_json = extract_and_clean_json(output_text)
#         parsed_data = json.loads(cleaned_json)

#         # Validate costs for all travelers/duration
#         total_cost = sum(
#             day.get('total_day_cost', 0) * traveler_count
#             for day in parsed_data.get('days', [])
#         )
#         if total_cost <= budget:
#             parsed_data['total_estimated_cost'] = total_cost
#             return parsed_data
#         else:
#             # Prune or scale so costs fit
#             return adjust_costs_to_budget(parsed_data, budget, traveler_count, duration)

#         # fallback if structure invalid
#         return generate_with_specific_places(user_inputs, destination)

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         return generate_with_specific_places(user_inputs, destination)

# def generate_itinerary_with_genai(user_inputs: dict, weather_forecasts: List[WeatherForecast] = None) -> dict:
#     model = get_genai_model()
#     destination = user_inputs.get('to_location') or user_inputs.get('location')
#     traveler_count = user_inputs.get('traveler_count', 1)
#     budget = user_inputs['budget']
#     duration = user_inputs['duration']
#     themes = user_inputs['themes']  # Now it's a list
#     start_date = user_inputs.get('start_date')
#     preferred_transport = user_inputs.get('preferred_transport', 'any')
#     user_comments = user_inputs.get('user_comments', '')

#     # Create themes string for prompt
#     themes_str = ", ".join(themes) if themes else "general"
#     primary_theme = themes[0] if themes else "cultural"

#     comments_info = ""
#     if user_comments:
#         comments_info = f"\n\nUSER PREFERENCES: \"{user_comments}\"\n" \
#                         f"Incorporate these preferences when selecting specific real places and activities."

#     budget_per_person = budget // traveler_count
#     budget_per_person_per_day = budget_per_person // duration
#     budget_per_activity_per_person = budget_per_person_per_day // 3

#     # Create weather context for prompt
#     weather_context = ""
#     if weather_forecasts and start_date:
#         weather_context = "\n\nWEATHER FORECAST:\n"
#         for i, forecast in enumerate(weather_forecasts):
#             weather_context += f"Day {i+1} ({forecast.date}): {forecast.condition}, Max Temp: {forecast.max_temp_c}°C, Min Temp: {forecast.min_temp_c}°C, Rain Chance: {forecast.chance_of_rain}%.\n"
#         weather_context += "Adjust activities and best_time based on weather (e.g., indoor activities for rain > 50% or temp > 35°C, morning for heat)."

#     prompt = f"""You are a travel expert for {destination}. Generate a {duration}-day itinerary with REAL, SPECIFIC places only.

# DESTINATION: {destination}
# THEMES: {themes_str} (focus on only selected themes throughout the trip)
# BUDGET: INR {budget} total, for {traveler_count} traveler(s)
# ALL ACTIVITY COSTS ARE PER TRAVELER, NOT TOTAL.
# DATES: {start_date if start_date else "Flexible"}
# TRANSPORT: {preferred_transport}
# {comments_info}
# {weather_context}

# IMPORTANT: Mix activities from only selected themes ({themes_str}) across different days. Don't focus on just one theme per day.

# RETURN ONLY VALID JSON - NO other text, explanations, or markdown:

# {{
#   "days": [
#     {{
#       "day": 1,
#       "activities": [
#         {{
#           "name": "REAL PLACE NAME",
#           "description": "...",
#           "latitude": 00.0000,
#           "longitude": 00.0000,
#           "estimated_cost": {budget_per_activity_per_person},
#           "duration_hours": 2.5,
#           "category": "one_of_selected_themes",
#           "best_time": "10:00 AM - 1:00 PM"
#         }}
#       ],
#       "total_day_cost": {budget_per_person_per_day}
#     }}
#   ],
#   "total_estimated_cost": {budget_per_person}
# }}
# CRITICAL RULES:
# 1. Use ONLY real, famous, specific places in {destination}
# 2. NO generic names like "Activity 1" or "Local Market"
# 3. Each activity cost is per person (multiply by traveler_count for day and trip totals)
# 4. Per traveler daily cost must not exceed INR {budget_per_person_per_day}
# 5. All days combined for all travelers must not exceed total budget INR {budget}
# 6. Include activities from only selected themes: {themes_str}
# 7. Categories should match selected themes: {themes_str}
# 8. Include at least 4-6 activities per day, if budget allows
# 9. Vary the theme categories across days for diversity
# 10. Adjust best_time based on weather (e.g., morning for heat, indoor for rain > 50%)
# """

#     try:
#         response = model.generate_content(
#             prompt,
#             generation_config={
#                 "temperature": 0.3,
#                 "top_p": 0.7,
#                 "candidate_count": 1,
#             },
#         )

#         output_text = response.text or ""
#         cleaned_json = extract_and_clean_json(output_text)
#         parsed_data = json.loads(cleaned_json)

#         # Validate costs for all travelers/duration
#         total_cost = sum(
#             day.get('total_day_cost', 0) * traveler_count
#             for day in parsed_data.get('days', [])
#         )
#         if total_cost <= budget:
#             parsed_data['total_estimated_cost'] = total_cost
#             return parsed_data
#         else:
#             # Prune or scale so costs fit
#             return adjust_costs_to_budget(parsed_data, budget, traveler_count, duration)

#         # Fallback if structure invalid
#         return generate_with_specific_places(user_inputs, destination)

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         return generate_with_specific_places(user_inputs, destination)
    
def extract_and_clean_json(text: str) -> str:
    """Enhanced JSON extraction with multiple strategies"""
    
    # Strategy 1: Extract from code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'{\s*"days".*?}(?=\s*$|\s*\n\s*[^}])',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_text = match.group(1) if '```' in pattern else match.group(0)
            cleaned = clean_json_string(json_text)
            try:
                json.loads(cleaned)  # Test if valid
                return cleaned
            except:
                continue
    
    # Strategy 2: Find JSON-like structure
    start_pos = text.find('{"days"')
    if start_pos == -1:
        start_pos = text.find('{')
    
    if start_pos != -1:
        # Find matching closing brace
        brace_count = 0
        end_pos = start_pos
        
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        json_candidate = text[start_pos:end_pos]
        cleaned = clean_json_string(json_candidate)
        try:
            json.loads(cleaned)
            return cleaned
        except:
            pass
    
    # Strategy 3: Return cleaned original text
    return clean_json_string(text)

def validate_enhanced_json_structure(data: dict, destination: str) -> bool:
    """Enhanced validation checking for real places"""
    if not isinstance(data, dict) or "days" not in data:
        return False
        
    if not isinstance(data["days"], list) or len(data["days"]) == 0:
        return False
    
    generic_terms = [
        "activity", "explore", "visit", "tour", "experience",
        "day 1", "day 2", "day 3", "local", "nearby"
    ]
    
    for day in data["days"]:
        if not isinstance(day, dict) or "activities" not in day:
            return False
            
        activities = day.get("activities", [])
        if len(activities) < 2 or len(activities) > 4:
            return False
            
        for activity in activities:
            if not isinstance(activity, dict):
                return False
                
            name = activity.get("name", "").lower()
            if not name or any(term in name for term in generic_terms):
                print(f"❌ Generic activity detected: {activity.get('name')}")
                return False
                
            # Check for required fields
            required_fields = ["name", "description", "estimated_cost", "duration_hours", "category"]
            if not all(field in activity for field in required_fields):
                return False
    
    return True

def adjust_costs_to_budget(data: dict, max_budget: int) -> dict:
    """Adjust costs to fit within budget while maintaining proportions"""
    current_total = data.get('total_estimated_cost', 0)
    if current_total <= max_budget:
        return data
    
    reduction_factor = (max_budget * 0.95) / current_total  # 95% of budget for safety
    
    for day in data.get('days', []):
        day_cost = 0
        for activity in day.get('activities', []):
            old_cost = activity.get('estimated_cost', 0)
            new_cost = int(old_cost * reduction_factor)
            activity['estimated_cost'] = max(50, new_cost)  # Minimum cost of 50
            day_cost += activity['estimated_cost']
        day['total_day_cost'] = day_cost
    
    # Recalculate total
    data['total_estimated_cost'] = sum(day.get('total_day_cost', 0) for day in data.get('days', []))
    
    return data

def generate_with_specific_places(user_inputs: dict, destination: str) -> dict:
    """Generate itinerary with hardcoded real places with strict budget control for all travelers"""
    print(f"🔄 Generating with specific places for {destination}")

    # Real places database - now considers multiple themes
    themes = user_inputs.get('themes', ['cultural'])
    places_db = get_real_places_for_destination(destination.lower(), themes)

    if not places_db:
        return create_fallback_itinerary(user_inputs)

    duration = user_inputs['duration']
    total_budget = user_inputs['budget']
    traveler_count = user_inputs.get('traveler_count', 1)

    print(f"📊 Budget Control: Total=₹{total_budget}, Duration={duration} days, Travelers={traveler_count}")
    print(f"🎯 Selected Themes: {', '.join(themes)}")

    # Rest of the function remains the same but now considers multiple themes
    base_daily_budget = total_budget // duration

    daily_budgets = []
    remaining_budget = total_budget

    for day_num in range(duration):
        if day_num == duration - 1:  # Last day gets remaining budget
            daily_budgets.append(remaining_budget)
        else:
            variation = 0.85 + (day_num * 0.1)  # 0.85, 0.95, 1.05, 1.15...
            min_daily = int(base_daily_budget * 0.5)
            day_budget = min(
                int(base_daily_budget * variation),
                remaining_budget - (duration - day_num - 1) * min_daily
            )
            day_budget = max(day_budget, min_daily)
            daily_budgets.append(day_budget)
            remaining_budget -= day_budget

    print(f"💰 Daily Budgets: {daily_budgets}, Total: ₹{sum(daily_budgets)}")

    itinerary = {
        "days": [],
        "total_estimated_cost": 0
    }

    import random
    available_places = places_db.copy()
    random.shuffle(available_places)

    used_places = set()
    actual_total_cost = 0

    for day_num in range(1, duration + 1):
        day_budget = daily_budgets[day_num - 1]
        min_cost_per_activity = 100 * traveler_count
        activities_per_day = min(3, max(2, day_budget // (800 * traveler_count)))

        print(f"🗓️ Day {day_num}: Budget=₹{day_budget}, Activities={activities_per_day}")

        day_places = []
        for place in available_places:
            if place["name"] not in used_places:
                day_places.append(place)
                used_places.add(place["name"])
                if len(day_places) >= activities_per_day:
                    break

        if len(day_places) < activities_per_day:
            remaining_needed = activities_per_day - len(day_places)
            available_places_copy = [
                p for p in available_places
                if p["name"] not in [dp["name"] for dp in day_places]
            ]
            for i, place in enumerate(available_places_copy[:remaining_needed]):
                day_places.append(place)

        activities = []
        remaining_day_budget = day_budget

        cost_percentages = [0.45, 0.35, 0.2]
        for i, place in enumerate(day_places):
            is_last_activity = (i == len(day_places) - 1)
            if is_last_activity:
                activity_cost = remaining_day_budget
            else:
                activity_cost = int(day_budget * cost_percentages[i])
                if activity_cost > remaining_day_budget:
                    activity_cost = remaining_day_budget
            activity_cost = max(min_cost_per_activity, activity_cost)

            per_person_cost = max(100, activity_cost // traveler_count)

            start_hour = 9 + (i * 3)
            end_hour = start_hour + 2

            # Enhanced description mentioning multiple themes
            theme_context = f"Perfect for your {', '.join(themes)} themed trip"
            
            activities.append({
                "name": place["name"],
                "description": f"{place['description']} {theme_context}.",
                "latitude": place["latitude"] + random.uniform(-0.005, 0.005),
                "longitude": place["longitude"] + random.uniform(-0.005, 0.005),
                "estimated_cost": per_person_cost,
                "duration_hours": 2.0 + random.uniform(0, 1),
                "category": place["category"],
                "best_time": f"{start_hour:02d}:00 {'AM' if start_hour < 12 else 'PM'} - {end_hour:02d}:00 {'AM' if end_hour < 12 else 'PM'}"
            })
            remaining_day_budget -= activity_cost

            if remaining_day_budget <= 0:
                break

        day_total_cost = sum(activity["estimated_cost"] for activity in activities) * traveler_count

        itinerary["days"].append({
            "day": day_num,
            "activities": activities,
            "total_day_cost": day_total_cost
        })

        actual_total_cost += day_total_cost
        print(f"✅ Day {day_num} Total: ₹{day_total_cost} (Budget: ₹{day_budget})")

    # Final budget adjustment if needed
    if actual_total_cost > total_budget:
        print(f"⚠️ Cost exceeded budget, adjusting: ₹{actual_total_cost} > ₹{total_budget}")
        reduction_factor = (total_budget * 0.95) / actual_total_cost
        adjusted_total = 0
        for day in itinerary["days"]:
            adjusted_day_cost = 0
            for activity in day["activities"]:
                original_cost = activity["estimated_cost"]
                activity["estimated_cost"] = max(100, int(original_cost * reduction_factor))
                adjusted_day_cost += activity["estimated_cost"] * traveler_count
            day["total_day_cost"] = adjusted_day_cost
            adjusted_total += adjusted_day_cost
        actual_total_cost = adjusted_total

    itinerary["total_estimated_cost"] = actual_total_cost

    print(f"🎯 Final Result: ₹{actual_total_cost} / ₹{total_budget} budget")
    print(f"📊 Daily Costs: {[day['total_day_cost'] for day in itinerary['days']]}")

    calculated_total = sum(day['total_day_cost'] for day in itinerary['days'])
    if calculated_total != actual_total_cost:
        print(f"⚠️ Math mismatch: {calculated_total} vs {actual_total_cost}")
        itinerary["total_estimated_cost"] = calculated_total

    return itinerary

def get_real_places_for_destination(destination: str, themes: List[str]) -> list:
    """Get real places for popular destinations with theme-based filtering"""
    
    places = {
        "chennai": [
            {"name": "Marina Beach", "description": "World's second longest urban beach with golden sands and sea breeze", "latitude": 13.0478, "longitude": 80.2838, "category": "nature"},
            {"name": "Kapaleeshwarar Temple", "description": "Ancient Dravidian temple dedicated to Lord Shiva in Mylapore", "latitude": 13.0338, "longitude": 80.2619, "category": "heritage"},
            {"name": "Fort St. George", "description": "Historic British fort housing a museum with colonial artifacts", "latitude": 13.0836, "longitude": 80.2876, "category": "heritage"},
            {"name": "San Thome Basilica", "description": "Neo-Gothic basilica built over the tomb of St. Thomas", "latitude": 13.0336, "longitude": 80.2799, "category": "cultural"},
            {"name": "Government Museum", "description": "Second oldest museum in India with rich archaeological collections", "latitude": 13.0678, "longitude": 80.2619, "category": "cultural"},
            {"name": "Besant Nagar Beach", "description": "Clean beach popular among locals with food stalls", "latitude": 13.0064, "longitude": 80.2669, "category": "nature"},
            {"name": "Mahabalipuram Shore Temple", "description": "UNESCO World Heritage rock-cut temple complex by the sea", "latitude": 12.6269, "longitude": 80.1992, "category": "heritage"},
            {"name": "DakshinaChitra", "description": "Living museum showcasing South Indian heritage and crafts", "latitude": 12.8851, "longitude": 80.2252, "category": "cultural"},
            {"name": "Guindy National Park", "description": "Urban national park with deer and diverse bird species", "latitude": 13.0067, "longitude": 80.2206, "category": "nature"},
            {"name": "Express Avenue Mall", "description": "Modern shopping mall with international and local brands", "latitude": 13.0569, "longitude": 80.2378, "category": "shopping"},
            {"name": "Parthasarathy Temple", "description": "Ancient Vaishnavite temple dedicated to Lord Krishna", "latitude": 13.0386, "longitude": 80.2569, "category": "heritage"},
            {"name": "Elliot's Beach", "description": "Quieter beach in Besant Nagar with cafes and clean environment", "latitude": 13.0064, "longitude": 80.2669, "category": "nature"},
            {"name": "Spencer Plaza", "description": "One of India's first shopping malls with diverse retail options", "latitude": 13.0627, "longitude": 80.2707, "category": "shopping"},
            {"name": "Santhome Cathedral", "description": "Historic cathedral with Portuguese colonial architecture", "latitude": 13.0336, "longitude": 80.2799, "category": "cultural"},
            {"name": "VGP Universal Kingdom", "description": "Popular amusement park with thrilling rides and entertainment", "latitude": 12.9618, "longitude": 80.2464, "category": "adventure"},
            {"name": "T. Nagar", "description": "Bustling shopping district famous for silk sarees and jewelry", "latitude": 13.0418, "longitude": 80.2341, "category": "shopping"},
            {"name": "Birla Planetarium", "description": "Modern planetarium offering astronomy shows and space exhibitions", "latitude": 13.0604, "longitude": 80.2548, "category": "cultural"},
            {"name": "Pulicat Lake", "description": "Scenic saltwater lagoon perfect for bird watching and boating", "latitude": 13.4167, "longitude": 80.0833, "category": "nature"},
            {"name": "Phoenix MarketCity", "description": "Premium shopping and entertainment destination", "latitude": 13.0434, "longitude": 80.2290, "category": "shopping"},
            {"name": "Cholamandal Artists' Village", "description": "India's largest artists' commune showcasing contemporary art", "latitude": 12.8851, "longitude": 80.2252, "category": "cultural"}
        ],
        "bangalore": [
            {"name": "Lalbagh Botanical Garden", "description": "240-acre botanical garden with diverse flora and glass house", "latitude": 12.9507, "longitude": 77.5848, "category": "nature"},
            {"name": "Bangalore Palace", "description": "Tudor-style palace with elegant architecture and royal artifacts", "latitude": 12.9982, "longitude": 77.5920, "category": "heritage"},
            {"name": "ISKCON Temple", "description": "Modern temple complex dedicated to Lord Krishna", "latitude": 13.0099, "longitude": 77.5518, "category": "cultural"},
            {"name": "Cubbon Park", "description": "Green lung of the city spanning 300 acres", "latitude": 12.9719, "longitude": 77.5937, "category": "nature"},
            {"name": "Tipu Sultan's Summer Palace", "description": "Historic wooden palace showcasing Indo-Islamic architecture", "latitude": 12.9591, "longitude": 77.5670, "category": "heritage"},
            {"name": "Commercial Street", "description": "Bustling shopping street famous for clothes and accessories", "latitude": 12.9833, "longitude": 77.6094, "category": "shopping"},
            {"name": "UB City Mall", "description": "Luxury shopping destination with premium brands", "latitude": 12.9719, "longitude": 77.5937, "category": "shopping"},
            {"name": "Nandi Hills", "description": "Hill station near Bangalore perfect for sunrise views", "latitude": 13.3703, "longitude": 77.6838, "category": "adventure"},
            {"name": "Vidhana Soudha", "description": "Imposing government building showcasing Neo-Dravidian architecture", "latitude": 12.9794, "longitude": 77.5912, "category": "heritage"},
            {"name": "Bannerghatta National Park", "description": "Wildlife sanctuary with tigers, lions and safari experiences", "latitude": 12.7957, "longitude": 77.5719, "category": "adventure"},
            {"name": "Bull Temple", "description": "16th-century temple with massive granite Nandi bull statue", "latitude": 12.9434, "longitude": 77.5847, "category": "heritage"},
            {"name": "Ulsoor Lake", "description": "Scenic lake in city center perfect for boating", "latitude": 12.9817, "longitude": 77.6094, "category": "nature"},
            {"name": "Biere Club", "description": "Popular nightlife destination with craft beers and live music", "latitude": 12.9716, "longitude": 77.5946, "category": "nightlife"},
            {"name": "Koshy's Restaurant", "description": "Iconic heritage restaurant serving authentic South Indian cuisine", "latitude": 12.9716, "longitude": 77.5946, "category": "food"}
        ],
        "mumbai": [
            {"name": "Gateway of India", "description": "Iconic arch monument overlooking the Arabian Sea", "latitude": 18.9220, "longitude": 72.8347, "category": "heritage"},
            {"name": "Marine Drive", "description": "Scenic coastal road known as Queen's Necklace", "latitude": 18.9441, "longitude": 72.8226, "category": "sightseeing"},
            {"name": "Chhatrapati Shivaji Terminus", "description": "UNESCO World Heritage railway station with Victorian architecture", "latitude": 18.9401, "longitude": 72.8353, "category": "heritage"},
            {"name": "Elephanta Caves", "description": "Ancient rock-cut cave temples on Elephanta Island", "latitude": 18.9633, "longitude": 72.9314, "category": "heritage"},
            {"name": "Juhu Beach", "description": "Popular beach destination with street food", "latitude": 19.0896, "longitude": 72.8656, "category": "nature"},
            {"name": "Crawford Market", "description": "Historic market for fresh produce and spices", "latitude": 18.9467, "longitude": 72.8342, "category": "shopping"},
            {"name": "Hanging Gardens", "description": "Terraced gardens on Malabar Hill with city views", "latitude": 18.9562, "longitude": 72.8052, "category": "nature"},
            {"name": "Haji Ali Dargah", "description": "Floating mosque accessible during low tide", "latitude": 18.9826, "longitude": 72.8089, "category": "cultural"},
            {"name": "Bandra-Worli Sea Link", "description": "Cable-stayed bridge connecting Bandra and Worli", "latitude": 19.0176, "longitude": 72.8562, "category": "sightseeing"},
            {"name": "Colaba Causeway", "description": "Shopping street known for handicrafts and souvenirs", "latitude": 18.9067, "longitude": 72.8147, "category": "shopping"},
            {"name": "Sanjay Gandhi National Park", "description": "Urban national park with Kanheri Caves", "latitude": 19.2147, "longitude": 72.9643, "category": "nature"},
            {"name": "Siddhivinayak Temple", "description": "Famous Ganesha temple visited by celebrities", "latitude": 19.0176, "longitude": 72.8562, "category": "cultural"},
            {"name": "Trishna Restaurant", "description": "Award-winning seafood restaurant with contemporary Indian cuisine", "latitude": 18.9220, "longitude": 72.8347, "category": "food"},
            {"name": "Aer Bar", "description": "Rooftop bar with stunning city skyline views", "latitude": 19.0176, "longitude": 72.8562, "category": "nightlife"}
        ],
        "delhi": [
            {"name": "Red Fort", "description": "Magnificent Mughal fortress complex and UNESCO World Heritage Site", "latitude": 28.6562, "longitude": 77.2410, "category": "heritage"},
            {"name": "India Gate", "description": "War memorial arch honoring Indian soldiers", "latitude": 28.6129, "longitude": 77.2295, "category": "heritage"},
            {"name": "Qutub Minar", "description": "Tallest brick minaret showcasing Indo-Islamic architecture", "latitude": 28.5245, "longitude": 77.1855, "category": "heritage"},
            {"name": "Lotus Temple", "description": "Bahai House of Worship with lotus-shaped architecture", "latitude": 28.5535, "longitude": 77.2588, "category": "cultural"},
            {"name": "Chandni Chowk", "description": "Historic market area famous for street food", "latitude": 28.6506, "longitude": 77.2334, "category": "shopping"},
            {"name": "Humayun's Tomb", "description": "Mughal tomb inspiring the Taj Mahal design", "latitude": 28.5933, "longitude": 77.2507, "category": "heritage"},
            {"name": "Akshardham Temple", "description": "Modern Hindu temple complex with cultural exhibitions", "latitude": 28.6127, "longitude": 77.2773, "category": "cultural"},
            {"name": "Lodhi Gardens", "description": "City park with medieval tombs and landscaped gardens", "latitude": 28.5918, "longitude": 77.2273, "category": "nature"},
            {"name": "Connaught Place", "description": "Central shopping and business district", "latitude": 28.6315, "longitude": 77.2167, "category": "shopping"},
            {"name": "Jama Masjid", "description": "Largest mosque in India built by Shah Jahan", "latitude": 28.6507, "longitude": 77.2334, "category": "heritage"},
            {"name": "Raj Ghat", "description": "Memorial to Mahatma Gandhi at his cremation site", "latitude": 28.6418, "longitude": 77.2482, "category": "cultural"},
            {"name": "National Museum", "description": "Premier museum showcasing Indian history and art", "latitude": 28.6118, "longitude": 77.2194, "category": "cultural"},
            {"name": "Karim's Restaurant", "description": "Legendary Mughlai restaurant serving authentic Delhi cuisine", "latitude": 28.6506, "longitude": 77.2334, "category": "food"},
            {"name": "Hauz Khas Village", "description": "Trendy area with boutiques, cafes, and vibrant nightlife", "latitude": 28.5494, "longitude": 77.1956, "category": "nightlife"}
        ],
        "tiruchirappalli": [
            {"name": "Sri Ranganathaswamy Temple", "description": "Largest functioning Hindu temple complex in the world", "latitude": 10.8626, "longitude": 78.7066, "category": "heritage"},
            {"name": "Rock Fort Temple", "description": "Historic fort and temple complex carved out of rock", "latitude": 10.8155, "longitude": 78.7047, "category": "heritage"},
            {"name": "Jambukeswarar Temple", "description": "Ancient Shiva temple representing the water element", "latitude": 10.8626, "longitude": 78.7066, "category": "heritage"},
            {"name": "Kallanai Dam", "description": "Ancient dam across River Kaveri built by Cholas", "latitude": 11.0017, "longitude": 79.0083, "category": "heritage"},
            {"name": "Government Museum", "description": "Museum showcasing Chola bronzes and sculptures", "latitude": 10.8231, "longitude": 78.6869, "category": "cultural"},
            {"name": "Butterfly Garden", "description": "Nature park with diverse butterfly species", "latitude": 10.7905, "longitude": 78.7047, "category": "nature"},
            {"name": "Mukkombu", "description": "Tourist spot with gardens and river views", "latitude": 10.8626, "longitude": 78.7066, "category": "nature"},
            {"name": "Puliyancholai Falls", "description": "Scenic waterfall in Kolli Hills near Trichy", "latitude": 11.3667, "longitude": 78.3333, "category": "adventure"},
            {"name": "Samayapuram Temple", "description": "Famous Mariamman temple known for worship rituals", "latitude": 10.9667, "longitude": 78.6333, "category": "cultural"}
        ]
    }
    
    # Get places for the destination
    destination_places = places.get(destination, [])
    
    # If specific destination not found, create generic places
    if not destination_places:
        destination_places = [
            {"name": f"Historic Fort {destination.title()}", "description": f"Ancient fortress showcasing the rich history of {destination}", "latitude": 12.9716, "longitude": 77.5946, "category": "heritage"},
            {"name": f"Central Temple {destination.title()}", "description": f"Important religious site reflecting {destination}'s spiritual heritage", "latitude": 12.9816, "longitude": 77.6046, "category": "cultural"},
            {"name": f"City Gardens {destination.title()}", "description": f"Beautiful botanical gardens in the heart of {destination}", "latitude": 12.9616, "longitude": 77.5846, "category": "nature"},
            {"name": f"Traditional Market {destination.title()}", "description": f"Local market offering authentic {destination} crafts and food", "latitude": 12.9516, "longitude": 77.5746, "category": "shopping"},
            {"name": f"Heritage Museum {destination.title()}", "description": f"Museum showcasing {destination}'s cultural artifacts", "latitude": 12.9416, "longitude": 77.5646, "category": "cultural"},
            {"name": f"Riverside Park {destination.title()}", "description": f"Scenic park along {destination}'s main waterway", "latitude": 12.9316, "longitude": 77.5546, "category": "nature"},
        ]
    
    # Enhanced theme mapping for multiple theme support
    theme_mapping = {
        "heritage": ["heritage", "cultural"],
        "cultural": ["cultural", "heritage"],
        "nature": ["nature"],
        "adventure": ["adventure", "nature"],
        "shopping": ["shopping"],
        "food": ["food", "cultural", "shopping"],
        "nightlife": ["nightlife", "cultural", "shopping"],
        "sightseeing": ["sightseeing", "heritage", "cultural"]
    }
    
    # Collect all preferred categories from all selected themes
    preferred_categories = []
    for theme in themes:
        preferred_categories.extend(theme_mapping.get(theme, [theme]))
    
    # Remove duplicates while preserving order
    preferred_categories = list(dict.fromkeys(preferred_categories))
    
    # Sort places by theme relevance - prioritize places matching selected themes
    themed_places = []
    other_places = []
    
    for place in destination_places:
        if place["category"] in preferred_categories:
            themed_places.append(place)
        else:
            other_places.append(place)
    
    # Return themed places first, then others for variety
    return themed_places + other_places

def create_fallback_itinerary(user_inputs: dict) -> dict:
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    location_coords = get_location_coordinates_dict(destination)
    traveler_count = user_inputs.get('traveler_count', 1)
    budget_per_day = user_inputs['budget'] // user_inputs['duration']
    cost_per_activity = budget_per_day // 3
    user_comments = user_inputs.get('user_comments', '')
    themes = user_inputs.get('themes', ['cultural'])
    
    # Try to incorporate user preferences in fallback
    activity_suffix = ""
    if user_comments:
        if any(word in user_comments.lower() for word in ['food', 'eat', 'cuisine', 'restaurant']):
            activity_suffix = " (Food Experience)"
        elif any(word in user_comments.lower() for word in ['history', 'historical', 'heritage', 'culture']):
            activity_suffix = " (Cultural Site)"
        elif any(word in user_comments.lower() for word in ['nature', 'park', 'outdoor', 'garden']):
            activity_suffix = " (Natural Attraction)"
    
    fallback_data = {
        "days": [],
        "total_estimated_cost": user_inputs['budget']
    }
    
    themes_str = ", ".join(themes)
    
    for day_num in range(1, user_inputs['duration'] + 1):
        day_activities = []
        for i in range(2):  # 2 activities per day
            # Cycle through themes for variety
            current_theme = themes[i % len(themes)]
            activity = {
                "name": f"Explore {destination}{activity_suffix} - Day {day_num} Activity {i + 1}",
                "description": f"Discover the {themes_str} aspects of {destination}" + 
                              (f", tailored to your preferences: {user_comments[:50]}..." if user_comments else "."),
                "latitude": location_coords['latitude'] + (i * 0.01),
                "longitude": location_coords['longitude'] + (i * 0.01),
                "estimated_cost": cost_per_activity,
                "duration_hours": 2.5,
                "category": current_theme if current_theme in ['sightseeing', 'food', 'adventure', 'cultural', 'shopping', 'nature', 'nightlife', 'heritage'] else 'sightseeing',
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
        "tiruchirappalli": {"latitude": 10.8155, "longitude": 78.7047},
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
# @app.post("/trip/generate-itinerary", response_model=Itinerary)
# async def generate_itinerary(req: TripRequest):
#     """Generate itinerary with enhanced user comments processing and multiple themes support"""
#     print(f"Generating itinerary with themes: {req.themes}")
#     print(f"User comments: {req.user_comments[:100] if req.user_comments else 'None'}...")
    
#     # Validate that themes is not empty
#     if not req.themes or len(req.themes) == 0:
#         raise HTTPException(status_code=400, detail="At least one theme must be selected")
    
#     itinerary_data = generate_itinerary_with_genai(req.dict())
#     if not itinerary_data or "days" not in itinerary_data:
#         raise HTTPException(status_code=500, detail="Failed to generate itinerary")

#     hotels = find_hotels_near_location(req.to_location or req.location) if req.to_location or req.location else []
#     route_details = get_route_details(req.from_location, req.to_location, req.preferred_transport or "driving") if req.from_location and req.to_location else None
#     weather_forecasts = get_weather_forecast(req.to_location or req.location, req.start_date, req.duration) if req.start_date else []

#     try:
#         days = []
#         start_date = datetime.strptime(req.start_date, "%Y-%m-%d") if req.start_date else datetime.now()
        
#         for idx, day_data in enumerate(itinerary_data.get("days", [])):
#             activities = []
#             for activity_data in day_data.get("activities", []):
#                 lat = activity_data.get("latitude")
#                 lng = activity_data.get("longitude")
#                 if lat is not None and lng is not None:
#                     try:
#                         lat = float(lat)
#                         lng = float(lng)
#                         if not validate_coordinates(lat, lng):
#                             lat, lng = None, None
#                     except (ValueError, TypeError):
#                         lat, lng = None, None
                
#                 cost = int(activity_data.get("estimated_cost", 0) * req.traveler_count) if activity_data.get("estimated_cost") else 0
#                 duration = float(activity_data.get("duration_hours")) if activity_data.get("duration_hours") else None
                
#                 activity = Activity(
#                     name=activity_data.get("name", "Unknown Activity"),
#                     description=activity_data.get("description", "No description available"),
#                     latitude=lat,
#                     longitude=lng,
#                     estimated_cost=cost,
#                     duration_hours=duration,
#                     category=activity_data.get("category", "general"),
#                     best_time=activity_data.get("best_time", "9:00 AM - 12:00 PM")
#                 )
#                 activities.append(activity)
            
#             day_cost = int(day_data.get("total_day_cost", 0) * req.traveler_count) if day_data.get("total_day_cost") else None
#             day = ItineraryDay(
#                 day=day_data.get("day", 0),
#                 date=(start_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
#                 activities=activities,
#                 weather=weather_forecasts[idx] if idx < len(weather_forecasts) else None,
#                 total_day_cost=day_cost
#             )
#             days.append(day)
        
#         total_cost = int(itinerary_data.get("total_estimated_cost", calculate_total_cost(days)))
#         if route_details and route_details.estimated_cost:
#             total_cost += route_details.estimated_cost
        
#         return Itinerary(
#             location=req.location,
#             duration=req.duration,
#             budget=req.budget,
#             themes=req.themes,  # Return the themes list
#             start_date=req.start_date,
#             traveler_count=req.traveler_count,
#             preferred_transport=req.preferred_transport,
#             from_location=req.from_location,
#             to_location=req.to_location,
#             user_comments=req.user_comments,
#             days=days,
#             total_estimated_cost=total_cost,
#             hotels=hotels,
#             route_details=route_details
#         )
#     except Exception as e:
#         print(f"Error parsing itinerary data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to parse itinerary data: {str(e)}")

# @app.post("/trip/generate-itinerary", response_model=Itinerary)
# async def generate_itinerary(req: TripRequest):
#     """Generate itinerary with correct cost calculations"""
    
#     # Validate that themes is not empty
#     if not req.themes or len(req.themes) == 0:
#         raise HTTPException(status_code=400, detail="At least one theme must be selected")
    
#     weather_forecasts = get_weather_forecast(req.to_location or req.location, req.start_date, req.duration) if req.start_date else []
    
#     itinerary_data = generate_itinerary_with_genai(req.model_dump(), weather_forecasts)
#     if not itinerary_data or "days" not in itinerary_data:
#         raise HTTPException(status_code=500, detail="Failed to generate itinerary")

#     hotels = find_hotels_near_location(req.to_location or req.location) if req.to_location or req.location else []
#     route_details = get_route_details(req.from_location, req.to_location, req.preferred_transport or "driving") if req.from_location and req.to_location else None

#     try:
#         days = []
#         start_date = datetime.strptime(req.start_date, "%Y-%m-%d") if req.start_date else datetime.now()
        
#         total_activities_cost = 0
        
#         for idx, day_data in enumerate(itinerary_data.get("days", [])):
#             activities = []
#             day_activities_cost = 0
            
#             for activity_data in day_data.get("activities", []):
#                 lat = activity_data.get("latitude")
#                 lng = activity_data.get("longitude")
#                 if lat is not None and lng is not None:
#                     try:
#                         lat = float(lat)
#                         lng = float(lng)
#                         if not validate_coordinates(lat, lng):
#                             lat, lng = None, None
#                     except (ValueError, TypeError):
#                         lat, lng = None, None
                
#                 # Cost from AI, assumed per person, convert to total for group
#                 cost_per_person_str = activity_data.get("estimated_cost", 0)
#                 cost_per_person = float(cost_per_person_str) if cost_per_person_str else 0
#                 total_cost = int(cost_per_person * req.traveler_count)
#                 duration = float(activity_data.get("duration_hours")) if activity_data.get("duration_hours") else None
                
#                 activity = Activity(
#                     name=activity_data.get("name", "Unknown Activity"),
#                     description=activity_data.get("description", "No description available"),
#                     latitude=lat,
#                     longitude=lng,
#                     estimated_cost=total_cost,  # Total for all travelers
#                     duration_hours=duration,
#                     category=activity_data.get("category", "general"),
#                     best_time=activity_data.get("best_time", "9:00 AM - 12:00 PM")
#                 )
#                 activities.append(activity)
#                 day_activities_cost += total_cost
            
#             # Day cost is total for all travelers
#             day_total_cost = day_activities_cost
#             total_activities_cost += day_activities_cost
            
#             day = ItineraryDay(
#                 day=day_data.get("day", 0),
#                 date=(start_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
#                 activities=activities,
#                 weather=weather_forecasts[idx] if idx < len(weather_forecasts) else None,
#                 total_day_cost=day_total_cost
#             )
#             days.append(day)
        
#         # Calculate cost breakdown
#         accommodation_cost = 0
#         if hotels and len(hotels) > 0:
#             avg_hotel_price = hotels[0].price_per_night if hasattr(hotels[0], 'price_per_night') and hotels[0].price_per_night else 1000
#             rooms_needed = max(1, (req.traveler_count + 1) // 2)  # Assume 2 travelers per room
#             accommodation_cost = avg_hotel_price * req.duration * rooms_needed
        
#         transportation_cost = route_details.estimated_cost if route_details and route_details.estimated_cost else 0
        
#         # Food estimate: 500 per person per day
#         food_cost = 500 * req.duration * req.traveler_count
        
#         # Miscellaneous: 10% of activities + accommodation + food
#         miscellaneous_cost = int((total_activities_cost + accommodation_cost + food_cost) * 0.1)
        
#         total_estimated_cost = total_activities_cost + accommodation_cost + transportation_cost + food_cost + miscellaneous_cost
                
#         cost_breakdown = {
#             "accommodation": accommodation_cost,
#             "activities": total_activities_cost,
#             "transportation": transportation_cost,
#             "food": food_cost,
#             "miscellaneous": miscellaneous_cost,
#             "total": total_estimated_cost
#         }
        
#         return Itinerary(
#             location=req.location,
#             duration=req.duration,
#             budget=req.budget,
#             themes=req.themes,
#             start_date=req.start_date,
#             traveler_count=req.traveler_count,
#             preferred_transport=req.preferred_transport,
#             from_location=req.from_location,
#             to_location=req.to_location,
#             user_comments=req.user_comments,
#             days=days,
#             total_estimated_cost=total_estimated_cost,
#             hotels=hotels,
#             route_details=route_details,
#             cost_breakdown=cost_breakdown
#         )
#     except Exception as e:
#         print(f"Error parsing itinerary data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to parse itinerary data: {str(e)}")
   
def find_hotels_near_location(location: str, radius: int = 5000) -> List[Hotel]:
    """Find hotels with estimated price per night"""
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
                # Estimate price per night based on price_level (0-4 scale)
                price_level = place.get('price_level', 2)  # Default to mid-range
                price_per_night = estimate_price_from_level(price_level)
                
                hotel = Hotel(
                    name=place.get('name', 'Unknown Hotel'),
                    address=place.get('vicinity', 'Address not available'),
                    rating=place.get('rating'),
                    price_level=price_level,
                    price_per_night=price_per_night,  # Add estimated price
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


def estimate_price_from_level(price_level: int) -> int:
    """
    Estimate price per night in INR based on Google's price_level (0-4)
    0 = Free, 1 = Inexpensive, 2 = Moderate, 3 = Expensive, 4 = Very Expensive
    """
    price_mapping = {
        0: 500,    # Free/Very Budget
        1: 1500,   # Inexpensive (₹1000-2000)
        2: 3000,   # Moderate (₹2500-4000)
        3: 6000,   # Expensive (₹5000-8000)
        4: 10000   # Very Expensive (₹8000+)
    }
    return price_mapping.get(price_level, 3000)  # Default to moderate


@app.post("/trip/generate-itinerary", response_model=Itinerary)
async def generate_itinerary(req: TripRequest):
    """Generate itinerary using Gemini's structured cost breakdown with hotel accommodation"""

    if not req.themes or len(req.themes) == 0:
        raise HTTPException(status_code=400, detail="At least one theme must be selected")

    weather_forecasts = get_weather_forecast(req.to_location or req.location, req.start_date, req.duration) if req.start_date else []
    
    # Get hotels before generating itinerary so we can pass them to AI
    hotels = find_hotels_near_location(req.to_location or req.location) if req.to_location or req.location else []
    
    # Calculate accommodation budget allocation (25-30% of total budget is typical)
    accommodation_budget = int(req.budget * 0.28)
    activities_and_other_budget = req.budget - accommodation_budget
    
    itinerary_data = generate_itinerary_with_genai(
        req.model_dump(), 
        weather_forecasts, 
        hotels, 
        accommodation_budget,
        activities_and_other_budget
    )

    if not itinerary_data or "days" not in itinerary_data:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary")

    route_details = get_route_details(req.from_location, req.to_location, req.preferred_transport or "driving") if req.from_location and req.to_location else None

    try:
        days = []
        start_date = datetime.strptime(req.start_date, "%Y-%m-%d") if req.start_date else datetime.now()
        total_activities_cost = 0

        for idx, day_data in enumerate(itinerary_data.get("days", [])):
            activities = []

            for activity_data in day_data.get("activities", []):
                lat, lng = None, None
                if "latitude" in activity_data and "longitude" in activity_data:
                    try:
                        lat = float(activity_data["latitude"])
                        lng = float(activity_data["longitude"])
                        if not validate_coordinates(lat, lng):
                            lat, lng = None, None
                    except Exception:
                        lat, lng = None, None

                # Cost is already total (per person * traveler_count from AI)
                total_cost = int(activity_data.get("estimated_cost", 0))
                duration = float(activity_data.get("duration_hours")) if activity_data.get("duration_hours") else None

                activity = Activity(
                    name=activity_data.get("name", "Unknown Activity"),
                    description=activity_data.get("description", "No description available"),
                    latitude=lat,
                    longitude=lng,
                    estimated_cost=total_cost,
                    duration_hours=duration,
                    category=activity_data.get("category", "general"),
                    best_time=activity_data.get("best_time", "9:00 AM - 12:00 PM")
                )

                activities.append(activity)
                total_activities_cost += total_cost

            # Calculate day total cost from daily breakdown
            daily_breakdown = day_data.get("daily_cost_breakdown", {})
            
            # Sum up all daily costs (activities, food, transportation, miscellaneous)
            day_total_cost = 0
            if daily_breakdown:
                day_total_cost = sum(
                    int(daily_breakdown.get(key, 0)) 
                    for key in ["activities"]
                )
            else:
                # Fallback: sum activity costs for this day
                day_total_cost = sum(a.estimated_cost for a in activities)

            day = ItineraryDay(
                day=day_data.get("day", idx + 1),
                date=(start_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
                activities=activities,
                weather=weather_forecasts[idx] if idx < len(weather_forecasts) else None,
                total_day_cost=day_total_cost
            )

            days.append(day)

        # Use Gemini's total cost breakdown
        cost_breakdown = itinerary_data.get("total_cost_breakdown", {})
        
        # Calculate accommodation cost based on hotels and duration
        accommodation_cost = 0
        if hotels and len(hotels) > 0:
            # Get average hotel price from available hotels
            avg_hotel_price = sum(h.price_per_night for h in hotels) / len(hotels)
            
            # Calculate rooms needed (assuming 2 people per room)
            rooms_needed = (req.traveler_count + 1) // 2
            
            # Total accommodation cost = rooms * nights * price per night
            nights = req.duration - 1 if req.duration > 1 else 1  # Usually n-1 nights for n days
            accommodation_cost = int(rooms_needed * nights * avg_hotel_price)
            
            # Cap accommodation at the allocated budget
            if accommodation_cost > accommodation_budget:
                accommodation_cost = accommodation_budget
        else:
            # Fallback if no hotels found - use default estimate
            nights = req.duration - 1 if req.duration > 1 else 1
            rooms_needed = (req.traveler_count + 1) // 2
            accommodation_cost = min(int(rooms_needed * nights * 3000), accommodation_budget)  # Default ₹3000/night
        
        # Update cost breakdown with calculated accommodation
        if cost_breakdown:
            cost_breakdown["accommodation"] = accommodation_cost
        else:
            cost_breakdown = {
                "accommodation": accommodation_cost,
                "activities": total_activities_cost,
                "food": 0,
                "transportation": 0,
                "miscellaneous": 0,
                "total": 0
            }
        
        # Recalculate total
        total_estimated_cost = sum(
            int(v) for k, v in cost_breakdown.items() 
            if k != "total" and isinstance(v, (int, float))
        )
        cost_breakdown["total"] = total_estimated_cost

        # Enforce budget cap - scale down proportionally if over budget
        if total_estimated_cost > req.budget:
            scale_factor = req.budget / total_estimated_cost
            
            # Scale all components proportionally
            for key in cost_breakdown:
                if key != "total" and isinstance(cost_breakdown[key], (int, float)):
                    cost_breakdown[key] = int(cost_breakdown[key] * scale_factor)
            
            # Recalculate total after scaling
            total_estimated_cost = sum(
                int(v) for k, v in cost_breakdown.items() 
                if k != "total" and isinstance(v, (int, float))
            )
            cost_breakdown["total"] = total_estimated_cost
            
            # Also scale accommodation cost for consistency
            accommodation_cost = cost_breakdown["accommodation"]

        return Itinerary(
            location=req.location,
            duration=req.duration,
            budget=req.budget,
            themes=req.themes,
            start_date=req.start_date,
            traveler_count=req.traveler_count,
            preferred_transport=req.preferred_transport,
            from_location=req.from_location,
            to_location=req.to_location,
            user_comments=req.user_comments,
            days=days,
            total_estimated_cost=total_estimated_cost,
            hotels=hotels,
            route_details=route_details,
            cost_breakdown=cost_breakdown
        )

    except Exception as e:
        print(f"❌ Error parsing itinerary data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse itinerary data: {str(e)}")


def generate_itinerary_with_genai(
    user_inputs: dict, 
    weather_forecasts: List[WeatherForecast] = None,
    hotels: List = None,
    accommodation_budget: int = 0,
    activities_budget: int = 0
) -> dict:
    """Generate structured itinerary JSON using Gemini (with hotel and cost breakdowns)"""
    model = get_genai_model()
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    traveler_count = user_inputs.get('traveler_count', 1)
    budget = user_inputs['budget']
    duration = user_inputs['duration']
    themes = user_inputs['themes']
    start_date = user_inputs.get('start_date')
    preferred_transport = user_inputs.get('preferred_transport', 'any')
    user_comments = user_inputs.get('user_comments', '')

    themes_str = ", ".join(themes) if themes else "general"
    comments_info = f"\n\nUSER PREFERENCES: \"{user_comments}\"" if user_comments else ""

    # Budget calculations
    activities_budget_per_day = activities_budget // duration if duration > 0 else 0
    activities_budget_per_day_per_person = activities_budget_per_day // traveler_count if traveler_count > 0 else 0

    weather_context = ""
    if weather_forecasts and start_date:
        weather_context = "\n\nWEATHER FORECAST:\n"
        for i, forecast in enumerate(weather_forecasts):
            weather_context += f"Day {i+1} ({forecast.date}): {forecast.condition}, Max Temp: {forecast.max_temp_c}°C, Min Temp: {forecast.min_temp_c}°C, Rain Chance: {forecast.chance_of_rain}%.\n"
        weather_context += "Adjust activities and best_time based on weather."

    # Hotel context
    hotel_context = ""
    if hotels and len(hotels) > 0:
        hotel_context = "\n\nAVAILABLE HOTELS:\n"
        for hotel in hotels[:5]:  # Show top 5 hotels
            hotel_context += f"- {hotel.name}: INR {hotel.price_per_night}/night (Rating: {hotel.rating}/5)\n"
        avg_hotel_price = sum(h.price_per_night for h in hotels) / len(hotels)
        rooms_needed = (traveler_count + 1) // 2
        nights = duration - 1 if duration > 1 else 1
        estimated_accommodation = int(rooms_needed * nights * avg_hotel_price)
        hotel_context += f"\nEstimated accommodation cost: INR {min(estimated_accommodation, accommodation_budget)} ({rooms_needed} room(s) × {nights} night(s))"

    # Prompt for Gemini
    prompt = f"""
You are a travel expert for {destination}. Generate a {duration}-day itinerary with real, specific places only.

DESTINATION: {destination}
THEMES: {themes_str}
TOTAL BUDGET: INR {budget} (for ALL {traveler_count} traveler(s) for entire {duration}-day trip)
BUDGET ALLOCATION:
- Accommodation: INR {accommodation_budget} (handled separately)
- Activities, Food, Transport, Misc: INR {activities_budget} (≈ INR {activities_budget_per_day}/day for all travelers)
DATES: {start_date if start_date else "Flexible"}
TRANSPORT: {preferred_transport}
{comments_info}
{weather_context}
{hotel_context}

Return ONLY valid JSON, no explanations.

IMPORTANT COST GUIDELINES:
1. ALL costs must be TOTAL costs for ALL {traveler_count} travelers combined (not per person)
2. estimated_cost in each activity = per_person_cost × {traveler_count}
3. Daily costs must account for all travelers
4. DO NOT include accommodation in daily_cost_breakdown or activities - it's calculated separately
5. Stay within INR {activities_budget} for all non-accommodation expenses

STRUCTURE:
{{
  "days": [
    {{
      "day": 1,
      "activities": [
        {{
          "name": "REAL PLACE NAME",
          "description": "Brief description",
          "latitude": 00.0000,
          "longitude": 00.0000,
          "estimated_cost": 800,
          "cost_breakdown": {{
            "entry_fee": 400,
            "transport": 200,
            "guide": 200
          }},
          "duration_hours": 2.5,
          "category": "theme_name",
          "best_time": "10:00 AM - 1:00 PM"
        }}
      ],
      "daily_cost_breakdown": {{
        "activities": 2400,
        "food": 1200,
        "transportation": 400,
        "miscellaneous": 200,
        "total_day_cost": 4200
      }}
    }}
  ],
  "total_cost_breakdown": {{
    "activities": 0,
    "food": 0,
    "transportation": 0,
    "miscellaneous": 0,
    "total": 0
  }}
}}

RULES:
1. Use only real places in {destination}
2. ALL costs are for ALL {traveler_count} travelers combined
3. Total non-accommodation cost ≤ INR {activities_budget}
4. Include 4-6 activities/day
5. DO NOT include accommodation in breakdown (handled separately)
6. Ensure cost_breakdown components sum to estimated_cost for each activity
7. Make total_cost_breakdown sum match sum of daily costs
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.7,
                "candidate_count": 1,
            },
        )

        output_text = response.text or ""
        cleaned_json = extract_and_clean_json(output_text)
        parsed_data = json.loads(cleaned_json)

        # Ensure cost breakdown structure exists
        if "total_cost_breakdown" not in parsed_data:
            parsed_data["total_cost_breakdown"] = {
                "activities": 0,
                "food": 0,
                "transportation": 0,
                "miscellaneous": 0,
                "total": 0
            }
        else:
            # Ensure all keys exist
            for key in ["activities", "food", "transportation", "miscellaneous", "total"]:
                if key not in parsed_data["total_cost_breakdown"]:
                    parsed_data["total_cost_breakdown"][key] = 0

        # Note: accommodation is NOT added here - it's calculated in the main function
        
        return parsed_data

    except Exception as e:
        print(f"❌ Error generating itinerary: {str(e)}")
        return {"days": [], "total_cost_breakdown": {}}  
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

# Enhanced endpoint for analyzing user comments with multiple themes support
@app.post("/trip/analyze-preferences")
async def analyze_user_preferences(user_comments: str, destination: str, themes: List[str] = None):
    """Analyze user comments and provide personalized suggestions for multiple themes"""
    if not user_comments.strip():
        return {"suggestions": [], "themes": themes or [], "message": "No comments provided for analysis"}
    
    model = get_genai_model()
    
    themes_context = f" considering the selected themes: {', '.join(themes)}" if themes else ""
    
    prompt = f"""
    Analyze these user travel preferences for {destination}{themes_context}:
    "{user_comments}"
    
    Provide suggestions in the following format:
    1. How well the user preferences align with selected themes
    2. Specific activity suggestions that match both preferences and themes
    3. Timing recommendations
    4. Budget considerations
    5. Travel tips based on their preferences
    
    Keep response concise and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        analysis = response.text if response else "Unable to analyze preferences"
        
        # Extract themes based on keywords (enhanced)
        detected_themes = []
        comment_lower = user_comments.lower()
        
        theme_keywords = {
            'food': ['food', 'eat', 'cuisine', 'restaurant', 'local dishes', 'cooking', 'chef', 'dining'],
            'cultural': ['history', 'historical', 'museum',  'culture', 'traditional', 'art', 'music'],
            'adventure': ['adventure', 'hiking', 'outdoor', 'sports', 'thrill', 'climbing', 'trekking'],
            'nature': ['nature', 'park', 'garden', 'wildlife', 'scenic', 'forest', 'mountains', 'beach'],
            'shopping': ['shopping', 'market', 'buy', 'souvenir', 'boutique', 'mall', 'crafts'],
            'nightlife': ['nightlife', 'bar', 'club', 'evening', 'night', 'drinks', 'party'],
            'heritage': ['heritage', 'ancient', 'monument', 'temple', 'palace', 'fort', 'archaeological'],
            'sightseeing': ['sightseeing', 'tourist', 'landmark', 'famous', 'iconic', 'visit', 'see']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in comment_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return {
            "analysis": analysis,
            "suggested_themes": detected_themes,
            "selected_themes": themes or [],
            "user_comments": user_comments,
            "destination": destination
        }
    except Exception as e:
        return {
            "error": f"Failed to analyze preferences: {str(e)}",
            "suggested_themes": ["cultural"],  # fallback
            "selected_themes": themes or [],
            "user_comments": user_comments,
            "destination": destination
        }

@app.get("/status")
async def status():
    return {
        "status": "AI Trip Planner with Multiple Themes Support is running",
        "bookings_count": len(bookings),
        "features": [
            "AI-powered itinerary generation",
            "Multiple themes selection support",
            "User comments and preferences integration",
            "2-3 activities per day with best timing",
            "Weather forecast integration",
            "Activity coordinates for mapping",
            "Cost estimation with traveler count support",
            "Hotel recommendations via Google Places API",
            "Route planning via Google Directions API",
            "Multi-transport mode support",
            "Personalized recommendations based on user feedback",
            "Theme-based activity filtering and prioritization",
            "Robust JSON parsing",
            "Fallback itinerary generation"
        ],
        "supported_themes": [
            "cultural", "adventure", "heritage", "nightlife", 
            "food", "nature", "shopping", "sightseeing"
        ],
        "supported_transport_modes": [
            "driving", "car", "walking", "transit",
            "public_transport", "bicycling", "bike"
        ],
        "version": "4.1"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Enhanced AI Trip Planner API with Multiple Themes Support",
        "description": "Generate personalized travel itineraries with AI, supporting multiple themes and user preferences",
        "endpoints": {
            "generate_itinerary": "/trip/generate-itinerary",
            "analyze_preferences": "/trip/analyze-preferences",
            "book_itinerary": "/trip/book-itinerary",
            "get_coordinates": "/trip/coordinates/{location}",
            "get_hotels": "/trip/hotels/{location}",
            "get_route": "/trip/route?from_location=X&to_location=Y&travel_mode=Z",
            "user_bookings": "/trip/bookings/{user_id}",
            "status": "/status"
        },
        "new_features": [
            "Multiple themes selection support",
            "Enhanced theme-based activity filtering",
            "User comments and preferences integration",
            "Personalized activity recommendations",
            "Enhanced AI prompts with multiple theme context",
            "Preference analysis with theme alignment",
            "Tailored activity descriptions for multiple themes",
            "Weather forecast integration",
            "Hotel recommendations from Google Places",
            "Route planning with multiple transport modes"
        ],
        "supported_themes": [
            "cultural", "adventure", "heritage", "nightlife", 
            "food", "nature", "shopping", "sightseeing"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
