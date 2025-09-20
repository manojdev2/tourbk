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

app = FastAPI(title="AI Trip Planner with User Comments Support")

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
    user_comments: Optional[str] = None  # New field for user feedback/preferences

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
    user_comments: Optional[str] = None  # Include user comments in response
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

# def clean_json_string(json_string: str) -> str:
#     if "```json" in json_string:
#         json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
#         if json_match:
#             json_string = json_match.group(1)
#     json_string = json_string.strip()
#     json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
#     json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
#     json_string = re.sub(r',\s*}', '}', json_string)
#     json_string = re.sub(r',\s*]', ']', json_string)
#     json_string = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', json_string)
#     # json_string = re.sub(r'(\w+)(\s*:)', r'"\1"\2', json_string)
#     # json_string = re.sub(r'""(\w+)""(\s*:)', r'"\1"\2', json_string)
#     return json_string

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

def generate_itinerary_with_genai(user_inputs: dict) -> dict:
    model = get_genai_model()
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    traveler_count = user_inputs.get('traveler_count', 1)
    budget = user_inputs['budget']
    duration = user_inputs['duration']
    theme = user_inputs['theme']
    start_date = user_inputs.get('start_date')
    preferred_transport = user_inputs.get('preferred_transport', 'any')
    user_comments = user_inputs.get('user_comments', '')

    comments_info = ""
    if user_comments:
        comments_info = f"\n\nUSER PREFERENCES: \"{user_comments}\"\n" \
                        f"Incorporate these preferences when selecting specific real places and activities."

    budget_per_person = budget // traveler_count
    budget_per_person_per_day = budget_per_person // duration
    budget_per_activity_per_person = budget_per_person_per_day // 3

    prompt = f"""You are a travel expert for {destination}. Generate a {duration}-day itinerary with REAL, SPECIFIC places only.

DESTINATION: {destination}
THEME: {theme}
BUDGET: INR {budget} total, for {traveler_count} traveler(s)
ALL ACTIVITY COSTS ARE PER TRAVELER, NOT TOTAL.
DATES: {start_date if start_date else "Flexible"}
TRANSPORT: {preferred_transport}
{comments_info}

RETURN ONLY VALID JSON - NO other text, explanations, or markdown:

{{
  "days": [
    {{
      "day": 1,
      "activities": [
        {{
          "name": "REAL PLACE NAME",
          "description": "...",
          "latitude": 00.0000,
          "longitude": 00.0000,
          "estimated_cost": {budget_per_activity_per_person},
          "duration_hours": 2.5,
          "category": "{theme}",
          "best_time": "10:00 AM - 1:00 PM"
        }}
      ],
      "total_day_cost": {budget_per_person_per_day}
    }}
  ],
  "total_estimated_cost": {budget_per_person}
}}
CRITICAL RULES:
1. Use ONLY real, famous, specific places in {destination}
2. NO generic names like "Activity 1" or "Local Market"
3. Each activity cost is per person (multiply by traveler_count for day and trip totals)
4. Per traveler daily cost must not exceed INR {budget_per_person_per_day}
5. All days combined for all travelers must not exceed total budget INR {budget}
6. Focus on {theme} theme
7. Categories: sightseeing, food, adventure, cultural, shopping, nature, nightlife, heritage
8. Include atlease 4-6 activities per day, if budget allows
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

        # Validate costs for all travelers/duration
        total_cost = sum(
            day.get('total_day_cost', 0) * traveler_count
            for day in parsed_data.get('days', [])
        )
        if total_cost <= budget:
            parsed_data['total_estimated_cost'] = total_cost
            return parsed_data
        else:
            # Prune or scale so costs fit
            return adjust_costs_to_budget(parsed_data, budget, traveler_count, duration)

        # fallback if structure invalid
        return generate_with_specific_places(user_inputs, destination)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return generate_with_specific_places(user_inputs, destination)


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

    # Real places database
    places_db = get_real_places_for_destination(destination.lower(), user_inputs['theme'])

    if not places_db:
        return create_fallback_itinerary(user_inputs)

    duration = user_inputs['duration']
    total_budget = user_inputs['budget']      # This is the TOTAL budget for ALL travelers
    traveler_count = user_inputs.get('traveler_count', 1)

    print(f"📊 Budget Control: Total=₹{total_budget}, Duration={duration} days, Travelers={traveler_count}")

    # Strict budget distribution - total budget across all days (for group)
    base_daily_budget = total_budget // duration

    # --- Daily budgets logic
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
        # Limit: 2-3 activities based on group budget per day
        min_cost_per_activity = 100 * traveler_count
        activities_per_day = min(3, max(2, day_budget // (800 * traveler_count)))

        print(f"🗓️ Day {day_num}: Budget=₹{day_budget}, Activities={activities_per_day}")

        day_places = []
        # Pick unique places each day
        for place in available_places:
            if place["name"] not in used_places:
                day_places.append(place)
                used_places.add(place["name"])
                if len(day_places) >= activities_per_day:
                    break

        # If we run out, cycle through unused for variety
        if len(day_places) < activities_per_day:
            remaining_needed = activities_per_day - len(day_places)
            available_places_copy = [
                p for p in available_places
                if p["name"] not in [dp["name"] for dp in day_places]
            ]
            for i, place in enumerate(available_places_copy[:remaining_needed]):
                day_places.append(place)

        # Distribute group day budget across activities
        activities = []
        remaining_day_budget = day_budget

        cost_percentages = [0.45, 0.35, 0.2]
        for i, place in enumerate(day_places):
            is_last_activity = (i == len(day_places) - 1)
            if is_last_activity:
                activity_cost = remaining_day_budget
            else:
                # Each cost will serve all travelers
                activity_cost = int(day_budget * cost_percentages[i])
                if activity_cost > remaining_day_budget:
                    activity_cost = remaining_day_budget
            activity_cost = max(min_cost_per_activity, activity_cost)

            # Evenly split among travelers for display
            per_person_cost = max(100, activity_cost // traveler_count)

            # Time slots
            start_hour = 9 + (i * 3)
            end_hour = start_hour + 2

            activities.append({
                "name": place["name"],
                "description": f"{place['description']} Perfect for day {day_num} of your {user_inputs['theme']} themed trip.",
                "latitude": place["latitude"] + random.uniform(-0.005, 0.005),
                "longitude": place["longitude"] + random.uniform(-0.005, 0.005),
                "estimated_cost": per_person_cost,
                "duration_hours": 2.0 + random.uniform(0, 1),
                "category": place["category"],
                "best_time": f"{start_hour:02d}:00 {'AM' if start_hour < 12 else 'PM'} - {end_hour:02d}:00 {'AM' if end_hour < 12 else 'PM'}"
            })
            remaining_day_budget -= activity_cost

            # Stop if group budget sold out
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

    # Final check - never exceed overall group budget
    if actual_total_cost > total_budget:
        print(f"⚠️ Cost exceeded budget, adjusting: ₹{actual_total_cost} > ₹{total_budget}")
        reduction_factor = (total_budget * 0.95) / actual_total_cost  # Use 95% to be safe
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

    # Double-check
    calculated_total = sum(day['total_day_cost'] for day in itinerary['days'])
    if calculated_total != actual_total_cost:
        print(f"⚠️ Math mismatch: {calculated_total} vs {actual_total_cost}")
        itinerary["total_estimated_cost"] = calculated_total

    return itinerary

def get_real_places_for_destination(destination: str, theme: str) -> list:
    """Get real places for popular destinations with more variety"""
    
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
        ],
        "bangalore": [
            {"name": "Lalbagh Botanical Garden", "description": "240-acre botanical garden with diverse flora and glass house", "latitude": 12.9507, "longitude": 77.5848, "category": "nature"},
            {"name": "Bangalore Palace", "description": "Tudor-style palace with elegant architecture and royal artifacts", "latitude": 12.9982, "longitude": 77.5920, "category": "heritage"},
            {"name": "ISKCON Temple", "description": "Modern temple complex dedicated to Lord Krishna", "latitude": 13.0099, "longitude": 77.5518, "category": "cultural"},
            {"name": "Cubbon Park", "description": "Green lung of the city spanning 300 acres", "latitude": 12.9719, "longitude": 77.5937, "category": "nature"},
            {"name": "Tipu Sultan's Summer Palace", "description": "Historic wooden palace showcasing Indo-Islamic architecture", "latitude": 12.9591, "longitude": 77.5670, "category": "heritage"},
            {"name": "Commercial Street", "description": "Bustling shopping street famous for clothes and accessories", "latitude": 12.9833, "longitude": 77.6094, "category": "shopping"},
            {"name": "UB City Mall", "description": "Luxury shopping destination with premium brands", "latitude": 12.9719, "longitude": 77.5937, "category": "shopping"},
            {"name": "Nandi Hills", "description": "Hill station near Bangalore perfect for sunrise views", "latitude": 13.3703, "longitude": 77.6838, "category": "nature"},
            {"name": "Vidhana Soudha", "description": "Imposing government building showcasing Neo-Dravidian architecture", "latitude": 12.9794, "longitude": 77.5912, "category": "heritage"},
            {"name": "Bannerghatta National Park", "description": "Wildlife sanctuary with tigers, lions and safari experiences", "latitude": 12.7957, "longitude": 77.5719, "category": "adventure"},
            {"name": "Bull Temple", "description": "16th-century temple with massive granite Nandi bull statue", "latitude": 12.9434, "longitude": 77.5847, "category": "heritage"},
            {"name": "Ulsoor Lake", "description": "Scenic lake in city center perfect for boating", "latitude": 12.9817, "longitude": 77.6094, "category": "nature"},
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
        ],
        "delhi": [
            {"name": "Red Fort", "description": "Magnificent Mughal fortress complex and UNESCO World Heritage Site", "latitude": 28.6562, "longitude": 77.2410, "category": "heritage"},
            {"name": "India Gate", "description": "War memorial arch honoring Indian soldiers", "latitude": 28.6129, "longitude": 77.2295, "category": "heritage"},
            {"name": "Qutub Minar", "description": "Tallest brick minaret showcasing Indo-Islamic architecture", "latitude": 28.5245, "longitude": 77.1855, "category": "heritage"},
            {"name": "Lotus Temple", "description": "Bahá'í House of Worship with lotus-shaped architecture", "latitude": 28.5535, "longitude": 77.2588, "category": "cultural"},
            {"name": "Chandni Chowk", "description": "Historic market area famous for street food", "latitude": 28.6506, "longitude": 77.2334, "category": "shopping"},
            {"name": "Humayun's Tomb", "description": "Mughal tomb inspiring the Taj Mahal design", "latitude": 28.5933, "longitude": 77.2507, "category": "heritage"},
            {"name": "Akshardham Temple", "description": "Modern Hindu temple complex with cultural exhibitions", "latitude": 28.6127, "longitude": 77.2773, "category": "cultural"},
            {"name": "Lodhi Gardens", "description": "City park with medieval tombs and landscaped gardens", "latitude": 28.5918, "longitude": 77.2273, "category": "nature"},
            {"name": "Connaught Place", "description": "Central shopping and business district", "latitude": 28.6315, "longitude": 77.2167, "category": "shopping"},
            {"name": "Jama Masjid", "description": "Largest mosque in India built by Shah Jahan", "latitude": 28.6507, "longitude": 77.2334, "category": "heritage"},
            {"name": "Raj Ghat", "description": "Memorial to Mahatma Gandhi at his cremation site", "latitude": 28.6418, "longitude": 77.2482, "category": "cultural"},
            {"name": "National Museum", "description": "Premier museum showcasing Indian history and art", "latitude": 28.6118, "longitude": 77.2194, "category": "cultural"},
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
            {"name": "Samayapuram Temple", "description": "Famous Mariamman temple known for worship rituals", "latitude": 10.9667, "longitude": 78.6333, "category": "cultural"},
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
    
    # Filter by theme if possible
    theme_mapping = {
        "heritage": ["heritage", "cultural"],
        "cultural": ["cultural", "heritage"],
        "nature": ["nature"],
        "adventure": ["adventure", "nature"],
        "shopping": ["shopping"],
        "food": ["cultural", "shopping"],
        "nightlife": ["cultural", "shopping"],
    }
    
    preferred_categories = theme_mapping.get(theme, ["cultural", "heritage", "nature"])
    
    # Sort places by theme relevance
    themed_places = [p for p in destination_places if p["category"] in preferred_categories]
    other_places = [p for p in destination_places if p["category"] not in preferred_categories]
    
    # Return themed places first, then others for variety
    return themed_places + other_places

def create_fallback_itinerary(user_inputs: dict) -> dict:
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    location_coords = get_location_coordinates_dict(destination)
    traveler_count = user_inputs.get('traveler_count', 1)
    budget_per_day = user_inputs['budget'] // user_inputs['duration']
    cost_per_activity = budget_per_day // 3
    user_comments = user_inputs.get('user_comments', '')
    
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
    
    for day_num in range(1, user_inputs['duration'] + 1):
        day_activities = []
        for i in range(2):  # 2 activities per day
            activity = {
                "name": f"Explore {destination}{activity_suffix} - Day {day_num} Activity {i + 1}",
                "description": f"Discover the {user_inputs['theme']} aspects of {destination}" + 
                              (f", tailored to your preferences: {user_comments[:50]}..." if user_comments else "."),
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
    """Generate itinerary with enhanced user comments processing"""
    print(f"Generating itinerary with user comments: {req.user_comments[:100] if req.user_comments else 'None'}...")
    
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
            user_comments=req.user_comments,  # Include user comments in response
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

# New endpoint for analyzing user comments and providing suggestions
@app.post("/trip/analyze-preferences")
async def analyze_user_preferences(user_comments: str, destination: str):
    """Analyze user comments and provide personalized suggestions"""
    if not user_comments.strip():
        return {"suggestions": [], "themes": [], "message": "No comments provided for analysis"}
    
    model = get_genai_model()
    
    prompt = f"""
    Analyze these user travel preferences for {destination}:
    "{user_comments}"
    
    Provide suggestions in the following format:
    1. Recommended themes (cultural, adventure, food, nature, etc.)
    2. Specific activity suggestions
    3. Timing recommendations
    4. Budget considerations
    5. Travel tips based on their preferences
    
    Keep response concise and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        analysis = response.text if response else "Unable to analyze preferences"
        
        # Extract themes based on keywords
        themes = []
        if any(word in user_comments.lower() for word in ['food', 'eat', 'cuisine', 'restaurant', 'local dishes']):
            themes.append('food')
        if any(word in user_comments.lower() for word in ['history', 'historical', 'museum', 'heritage', 'culture']):
            themes.append('cultural')
        if any(word in user_comments.lower() for word in ['adventure', 'hiking', 'outdoor', 'sports', 'thrill']):
            themes.append('adventure')
        if any(word in user_comments.lower() for word in ['nature', 'park', 'garden', 'wildlife', 'scenic']):
            themes.append('nature')
        if any(word in user_comments.lower() for word in ['shopping', 'market', 'buy', 'souvenir']):
            themes.append('shopping')
        if any(word in user_comments.lower() for word in ['nightlife', 'bar', 'club', 'evening', 'night']):
            themes.append('nightlife')
        
        return {
            "analysis": analysis,
            "suggested_themes": themes,
            "user_comments": user_comments,
            "destination": destination
        }
    except Exception as e:
        return {
            "error": f"Failed to analyze preferences: {str(e)}",
            "suggested_themes": ["cultural"],  # fallback
            "user_comments": user_comments,
            "destination": destination
        }

@app.get("/status")
async def status():
    return {
        "status": "AI Trip Planner with User Comments Support is running",
        "bookings_count": len(bookings),
        "features": [
            "AI-powered itinerary generation",
            "User comments and preferences integration",
            "2-3 activities per day with best timing",
            "Weather forecast integration",
            "Activity coordinates for mapping",
            "Cost estimation with traveler count support",
            "Hotel recommendations via Google Places API",
            "Route planning via Google Directions API",
            "Multi-transport mode support",
            "Personalized recommendations based on user feedback",
            "Theme-based recommendations",
            "Robust JSON parsing",
            "Fallback itinerary generation"
        ],
        "supported_transport_modes": [
            "driving", "car", "walking", "transit",
            "public_transport", "bicycling", "bike"
        ],
        "version": "4.0"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Enhanced AI Trip Planner API with User Comments Support",
        "description": "Generate personalized travel itineraries with AI, incorporating user preferences and comments",
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
            "User comments and preferences integration",
            "Personalized activity recommendations",
            "Enhanced AI prompts with user context",
            "Preference analysis endpoint",
            "Tailored activity descriptions",
            "Weather forecast integration",
            "Hotel recommendations from Google Places",
            "Route planning with multiple transport modes"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)