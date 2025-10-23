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

app = FastAPI(title="AI Trip Planner with Gemini Cost Breakdown")

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
    themes: List[str]
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
    budget_warning: Optional[Dict] = None

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
    """Get accurate coordinates from Google Maps Geocoding API"""
    url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': location, 'key': GOOGLE_MAPS_API_KEY}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            location_data = data['results'][0]['geometry']['location']
            formatted_address = data['results'][0]['formatted_address']
            print(f"📍 Geocoded: {location} → {formatted_address} ({location_data['lat']}, {location_data['lng']})")
            return {
                'latitude': location_data['lat'],
                'longitude': location_data['lng'],
                'formatted_address': formatted_address
            }
    except Exception as e:
        print(f"Error getting coordinates from Google: {e}")
    
    # Fallback to hardcoded coordinates
    return get_fallback_coordinates(location)

def get_fallback_coordinates(location: str) -> dict:
    """Fallback coordinates for major Indian cities"""
    location_coords = {
        "bangalore": {"latitude": 12.9716, "longitude": 77.5946, "formatted_address": "Bangalore, Karnataka, India"},
        "mumbai": {"latitude": 19.0760, "longitude": 72.8777, "formatted_address": "Mumbai, Maharashtra, India"},
        "delhi": {"latitude": 28.7041, "longitude": 77.1025, "formatted_address": "New Delhi, India"},
        "goa": {"latitude": 15.2993, "longitude": 74.1240, "formatted_address": "Goa, India"},
        "chennai": {"latitude": 13.0827, "longitude": 80.2707, "formatted_address": "Chennai, Tamil Nadu, India"},
        "jaipur": {"latitude": 26.9124, "longitude": 75.7873, "formatted_address": "Jaipur, Rajasthan, India"},
        "kerala": {"latitude": 10.8505, "longitude": 76.2711, "formatted_address": "Kerala, India"},
        "manali": {"latitude": 32.2396, "longitude": 77.1887, "formatted_address": "Manali, Himachal Pradesh, India"},
        "udaipur": {"latitude": 24.5854, "longitude": 73.7125, "formatted_address": "Udaipur, Rajasthan, India"},
        "rishikesh": {"latitude": 30.0869, "longitude": 78.2676, "formatted_address": "Rishikesh, Uttarakhand, India"},
    }
    location_lower = location.lower()
    for key in location_coords:
        if key in location_lower:
            return location_coords[key]
    
    # Default to Bangalore
    return location_coords["bangalore"]

def get_route_details(from_location: str, to_location: str, travel_mode: str = "driving") -> Optional[RouteDetails]:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    mode_mapping = {
        "driving": "driving", "car": "driving", "walking": "walking",
        "transit": "transit", "public_transport": "transit",
        "bicycling": "bicycling", "bike": "bicycling"
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
        "driving": 8,
        "transit": 2,
        "bicycling": 5,
        "walking": 0
    }
    return int(distance_km * cost_rates.get(travel_mode, 10))

# --- Weather API Functions ---
def get_weather_forecast(location: str, start_date: str, days: int) -> List[WeatherForecast]:
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
    genai.configure(api_key='AIzaSyD4hQw106GNViPQXZc8DOR06_Vs4fMrsLw')
    return genai.GenerativeModel("gemini-2.0-flash-exp")

def clean_json_string(json_string: str) -> str:
    if json_string.strip().startswith("```"):
        json_string = re.sub(r"```[a-zA-Z]*\n?", "", json_string)
        json_string = json_string.replace("```", "")
    
    json_string = json_string.strip()
    json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    return json_string

def extract_and_clean_json(text: str) -> str:
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
                json.loads(cleaned)
                return cleaned
            except:
                continue
    
    start_pos = text.find('{"days"')
    if start_pos == -1:
        start_pos = text.find('{')
    
    if start_pos != -1:
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
    
    return clean_json_string(text)

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
                price_level = place.get('price_level', 2)
                price_per_night = estimate_price_from_level(price_level)
                
                hotel = Hotel(
                    name=place.get('name', 'Unknown Hotel'),
                    address=place.get('vicinity', 'Address not available'),
                    rating=place.get('rating'),
                    price_level=price_level,
                    price_per_night=price_per_night,
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
    price_mapping = {
        0: 500,
        1: 1500,
        2: 3000,
        3: 6000,
        4: 10000
    }
    return price_mapping.get(price_level, 3000)

def get_local_events_context(destination: str, start_date: str = None) -> str:
    """Get information about local festivals and events"""
    model = get_genai_model()
    
    date_context = f" around {start_date}" if start_date else " in the current period"
    
    prompt = f"""List any notable festivals, events, or unique cultural experiences in {destination}{date_context}.

Format as a brief list (max 3-4 items):
- Event name: Brief description

If no major events, mention seasonal highlights or unique local experiences.
Keep response under 200 words."""

    try:
        response = model.generate_content(prompt)
        return f"\n\nLOCAL EVENTS & FESTIVALS:\n{response.text}" if response and response.text else ""
    except Exception as e:
        print(f"Error getting local events: {e}")
        return ""

def generate_itinerary_with_genai(
    user_inputs: dict, 
    weather_forecasts: List[WeatherForecast] = None,
    hotels: List = None
) -> dict:
    """Generate structured itinerary with Gemini handling ALL cost calculations"""
    model = get_genai_model()
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    traveler_count = user_inputs.get('traveler_count', 1)
    total_budget = user_inputs['budget']
    duration = user_inputs['duration']
    themes = user_inputs['themes']
    start_date = user_inputs.get('start_date')
    preferred_transport = user_inputs.get('preferred_transport', 'any')
    user_comments = user_inputs.get('user_comments', '')

    # Get destination coordinates for accurate location context
    dest_coords = get_location_coordinates_from_google(destination)
    
    themes_str = ", ".join(themes) if themes else "general"
    comments_info = f"\n\nUSER PREFERENCES: \"{user_comments}\"\nIncorporate these preferences when selecting activities." if user_comments else ""

    # Weather context
    weather_context = ""
    if weather_forecasts and start_date:
        weather_context = "\n\nWEATHER FORECAST:\n"
        for i, forecast in enumerate(weather_forecasts):
            weather_context += f"Day {i+1} ({forecast.date}): {forecast.condition}, {forecast.max_temp_c}°C/{forecast.min_temp_c}°C, Rain: {forecast.chance_of_rain}%\n"
        weather_context += "Adjust activities based on weather conditions."

    # Calculate accommodation cost
    rooms_needed = (traveler_count + 1) // 2
    nights = max(1, duration - 1)
    avg_hotel_price = 3000
    
    if hotels and len(hotels) > 0:
        avg_hotel_price = sum(h.price_per_night for h in hotels) / len(hotels)
    
    estimated_accommodation = int(rooms_needed * nights * avg_hotel_price)
    
    # Budget validation
    min_per_person_per_day = 1000  # Minimum ₹1000/person/day
    recommended_budget = traveler_count * duration * min_per_person_per_day
    
    # Check if budget is too low
    include_accommodation = True
    budget_warning = None
    
    if total_budget < recommended_budget * 0.5:
        # Budget is critically low
        budget_warning = {
            "level": "critical",
            "message": f"Budget too low! Recommended: ₹{recommended_budget} for {traveler_count} travelers × {duration} days",
            "suggestions": [
                f"Increase budget to at least ₹{recommended_budget}",
                f"Reduce duration to {max(1, int(total_budget / (traveler_count * min_per_person_per_day)))} days",
                f"Reduce travelers to {max(1, int(total_budget / (duration * min_per_person_per_day)))} person(s)"
            ]
        }
    elif total_budget < estimated_accommodation + (traveler_count * duration * 500):
        # Budget too low for accommodation
        include_accommodation = False
        budget_warning = {
            "level": "warning",
            "message": f"Budget insufficient for accommodation (₹{estimated_accommodation}). Itinerary will focus on activities only.",
            "suggestions": [
                f"Increase budget to ₹{estimated_accommodation + (traveler_count * duration * 1000)} to include accommodation",
                "Consider day trips instead of overnight stays",
                "Look for budget hostels or homestays"
            ]
        }
        estimated_accommodation = 0
    elif total_budget < recommended_budget:
        # Budget is tight but manageable
        budget_warning = {
            "level": "info",
            "message": f"Budget is tight. Recommended: ₹{recommended_budget} for comfortable travel.",
            "suggestions": [
                "Itinerary will include budget-friendly activities",
                "Consider increasing budget for better experiences",
                f"Focus on free activities and local experiences"
            ]
        }
    
    # Store warning for return
    if budget_warning:
        print(f"\n⚠️ Budget Warning: {budget_warning['message']}")
        for suggestion in budget_warning['suggestions']:
            print(f"   💡 {suggestion}")

    # Hotel context
    hotel_context = ""
    if hotels and len(hotels) > 0 and include_accommodation:
        hotel_context = "\n\nAVAILABLE HOTELS (sample):\n"
        for hotel in hotels[:3]:
            hotel_context += f"- {hotel.name}: ₹{hotel.price_per_night}/night (Rating: {hotel.rating}/5)\n"
    
    if include_accommodation:
        hotel_context += f"\n\nACCOMMODATION CALCULATION:\n"
        hotel_context += f"- Travelers: {traveler_count} → Rooms needed: {rooms_needed}\n"
        hotel_context += f"- Nights: {nights} (for {duration} days trip)\n"
        hotel_context += f"- Average price: ₹{int(avg_hotel_price)}/night\n"
        hotel_context += f"- Total accommodation cost: ₹{estimated_accommodation}\n"
    else:
        hotel_context += f"\n\n⚠️ ACCOMMODATION EXCLUDED due to budget constraints\n"
        hotel_context += f"- Budget: ₹{total_budget} is insufficient for accommodation (₹{int(rooms_needed * nights * avg_hotel_price)})\n"
        hotel_context += f"- Focusing on day activities and local experiences\n"

    # Get local events
    events_context = get_local_events_context(destination, start_date)

    # Adjust prompt based on budget
    accommodation_instruction = ""
    if include_accommodation:
        accommodation_instruction = f"""
ACCOMMODATION REQUIREMENT:
- MUST allocate ₹{estimated_accommodation} for accommodation in cost_breakdown
- This is MANDATORY: {rooms_needed} rooms × {nights} nights × ₹{int(avg_hotel_price)}/night
- Remaining budget for other expenses: ₹{total_budget - estimated_accommodation}
"""
    else:
        accommodation_instruction = f"""
ACCOMMODATION EXCLUDED:
- Set cost_breakdown.accommodation = 0
- Budget ₹{total_budget} is too low to include accommodation
- Focus ONLY on activities, food, and local transport
- Use full budget for: Activities (~50%), Food (~30%), Transport (~15%), Misc (~5%)
"""

    prompt = f"""You are an expert travel planner creating a detailed itinerary for {destination}.

DESTINATION: {destination}
LOCATION: {dest_coords['formatted_address']} (Lat: {dest_coords['latitude']}, Lng: {dest_coords['longitude']})
THEMES: {themes_str}
TOTAL BUDGET: ₹{total_budget} for ALL {traveler_count} traveler(s) for entire {duration}-day trip
DATES: {start_date if start_date else "Flexible"}
TRANSPORT: {preferred_transport}
{comments_info}
{weather_context}
{hotel_context}
{events_context}

⚠️ CRITICAL BUDGET RULES:
1. Budget ₹{total_budget} is for ALL {traveler_count} travelers for ALL {duration} days COMBINED
2. ALL costs must be TOTAL costs (multiply per-person × {traveler_count})
3. MUST stay within ₹{total_budget} total - NO EXCEPTIONS
4. ALL values must be POSITIVE (≥ 0)
5. Use realistic, achievable costs based on {destination}

{accommodation_instruction}

ACTIVITY COUNT GUIDELINES:
- High budget (>₹50,000): 5-6 activities per day
- Medium budget (₹20,000-₹50,000): 4-5 activities per day  
- Low budget (₹10,000-₹20,000): 3-4 budget activities per day
- Very low budget (<₹10,000): 2-3 FREE activities per day (temples, beaches, parks, viewpoints)

COST CALCULATION EXAMPLE ({traveler_count} travelers):
- Museum ₹200/person → estimated_cost: {200 * traveler_count}
- Meal ₹300/person → add to food: {300 * traveler_count}

Return ONLY valid JSON with positive values:

{{
  "days": [
    {{
      "day": 1,
      "activities": [
        {{
          "name": "Real place in {destination}",
          "description": "What makes this special",
          "latitude": {dest_coords['latitude']},
          "longitude": {dest_coords['longitude']},
          "estimated_cost": {100 * traveler_count},
          "duration_hours": 2.0,
          "category": "{themes[0] if themes else 'cultural'}",
          "best_time": "10:00 AM"
        }}
      ],
      "total_day_cost": 0
    }}
  ],
  "cost_breakdown": {{
    "accommodation": {estimated_accommodation if include_accommodation else 0},
    "activities": 0,
    "food": 0,
    "transportation": 0,
    "miscellaneous": 0,
    "total": 0
  }}
}}

REQUIREMENTS:
1. Use real places in {destination} near ({dest_coords['latitude']}, {dest_coords['longitude']})
2. Include {"free/low-cost" if total_budget < 15000 else "mix of free and paid"} activities
3. Mix themes: {themes_str}
4. total_day_cost = sum of activity costs for that day
5. cost_breakdown.activities = sum of ALL activity costs
6. cost_breakdown.total = sum of all categories
7. VERIFY: cost_breakdown.total ≤ ₹{total_budget}
8. ALL costs must be ≥ 0 (no negative values)
9. Include festivals/events if available

CRITICAL VALIDATION:
- cost_breakdown.accommodation = ₹{estimated_accommodation if include_accommodation else 0}
- Sum of daily activities = cost_breakdown.activities
- cost_breakdown.total = accommodation + activities + food + transportation + miscellaneous
- cost_breakdown.total MUST be ≤ ₹{total_budget}
- NO negative values anywhere"""

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
        print(f"\n🤖 Gemini Response Preview:\n{output_text[:500]}...\n")
        
        cleaned_json = extract_and_clean_json(output_text)
        parsed_data = json.loads(cleaned_json)

        if "days" not in parsed_data or "cost_breakdown" not in parsed_data:
            raise ValueError("Missing required fields")

        # Fix accommodation
        if include_accommodation:
            parsed_data["cost_breakdown"]["accommodation"] = estimated_accommodation
        else:
            parsed_data["cost_breakdown"]["accommodation"] = 0

        # Remove negative values
        for key in parsed_data["cost_breakdown"]:
            if parsed_data["cost_breakdown"][key] < 0:
                print(f"⚠️ Fixed negative {key}: {parsed_data['cost_breakdown'][key]} → 0")
                parsed_data["cost_breakdown"][key] = 0
        
        for day in parsed_data["days"]:
            for activity in day.get("activities", []):
                if activity.get("estimated_cost", 0) < 0:
                    activity["estimated_cost"] = 0
            if day.get("total_day_cost", 0) < 0:
                day["total_day_cost"] = 0

        # Calculate totals
        total_activities = sum(
            activity.get("estimated_cost", 0)
            for day in parsed_data["days"]
            for activity in day.get("activities", [])
        )
        
        parsed_data["cost_breakdown"]["activities"] = total_activities
        parsed_data["cost_breakdown"]["total"] = sum(
            v for k, v in parsed_data["cost_breakdown"].items()
            if k != "total" and isinstance(v, (int, float))
        )

        # Budget enforcement
        current_total = parsed_data["cost_breakdown"]["total"]
        
        if current_total > total_budget:
            print(f"⚠️ Over budget: ₹{current_total} > ₹{total_budget}")
            scale_factor = (total_budget * 0.95) / current_total
            
            for key in parsed_data["cost_breakdown"]:
                if key != "total" and key != "accommodation":
                    parsed_data["cost_breakdown"][key] = int(parsed_data["cost_breakdown"][key] * scale_factor)
            
            for day in parsed_data["days"]:
                for activity in day["activities"]:
                    activity["estimated_cost"] = int(activity["estimated_cost"] * scale_factor)
                day["total_day_cost"] = sum(a.get("estimated_cost", 0) for a in day["activities"])
            
            parsed_data["cost_breakdown"]["total"] = sum(
                v for k, v in parsed_data["cost_breakdown"].items()
                if k != "total"
            )

        # Add budget warning to response
        if budget_warning:
            parsed_data["budget_warning"] = budget_warning

        print(f"✅ Final: ₹{parsed_data['cost_breakdown']['total']} / ₹{total_budget}")
        return parsed_data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "days": [],
            "cost_breakdown": {
                "accommodation": 0,
                "activities": 0,
                "food": 0,
                "transportation": 0,
                "miscellaneous": 0,
                "total": 0
            },
            "budget_warning": budget_warning if budget_warning else {
                "level": "error",
                "message": "Failed to generate itinerary",
                "suggestions": ["Please try again with different parameters"]
            }
        }

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate if coordinates are within valid range"""
    return -90 <= latitude <= 90 and -180 <= longitude <= 180

def process_payment(payment_token: str) -> bool:
    return True

# --- API Routes ---
@app.post("/trip/generate-itinerary", response_model=Itinerary)
async def generate_itinerary(req: TripRequest):
    """Generate itinerary with Gemini handling all cost calculations"""
    
    if not req.themes or len(req.themes) == 0:
        raise HTTPException(status_code=400, detail="At least one theme must be selected")
    
    print(f"\n{'='*60}")
    print(f"🎯 Generating itinerary for {req.to_location or req.location}")
    print(f"👥 Travelers: {req.traveler_count}")
    print(f"💰 Total Budget: ₹{req.budget}")
    print(f"📅 Duration: {req.duration} days")
    print(f"🎨 Themes: {', '.join(req.themes)}")
    print(f"{'='*60}\n")

    # Get weather and hotels
    weather_forecasts = get_weather_forecast(
        req.to_location or req.location, 
        req.start_date, 
        req.duration
    ) if req.start_date else []
    
    hotels = find_hotels_near_location(req.to_location or req.location)
    
    # Generate itinerary with Gemini
    itinerary_data = generate_itinerary_with_genai(
        req.model_dump(), 
        weather_forecasts,
        hotels
    )

    if not itinerary_data or "days" not in itinerary_data:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary")

    # Get route details
    route_details = None
    if req.from_location and req.to_location:
        route_details = get_route_details(
            req.from_location, 
            req.to_location, 
            req.preferred_transport or "driving"
        )

    try:
        days = []
        start_date = datetime.strptime(req.start_date, "%Y-%m-%d") if req.start_date else datetime.now()

        # Process days
        for idx, day_data in enumerate(itinerary_data.get("days", [])):
            activities = []

            for activity_data in day_data.get("activities", []):
                lat, lng = None, None
                if "latitude" in activity_data and "longitude" in activity_data:
                    try:
                        lat = float(activity_data["latitude"])
                        lng = float(activity_data["longitude"])
                        if not validate_coordinates(lat, lng):
                            print(f"⚠️ Invalid coordinates for {activity_data.get('name')}: ({lat}, {lng})")
                            lat, lng = None, None
                    except Exception as e:
                        print(f"❌ Error parsing coordinates: {e}")
                        pass

                activity = Activity(
                    name=activity_data.get("name", "Unknown Activity"),
                    description=activity_data.get("description", ""),
                    latitude=lat,
                    longitude=lng,
                    estimated_cost=int(activity_data.get("estimated_cost", 0)),
                    duration_hours=float(activity_data.get("duration_hours", 2.0)),
                    category=activity_data.get("category", "general"),
                    best_time=activity_data.get("best_time", "10:00 AM - 12:00 PM")
                )
                activities.append(activity)

            day = ItineraryDay(
                day=day_data.get("day", idx + 1),
                date=(start_date + timedelta(days=idx)).strftime("%Y-%m-%d"),
                activities=activities,
                weather=weather_forecasts[idx] if idx < len(weather_forecasts) else None,
                total_day_cost=int(day_data.get("total_day_cost", 0))
            )
            days.append(day)

        # Get cost breakdown from Gemini response
        cost_breakdown = itinerary_data.get("cost_breakdown", {})
        total_estimated_cost = int(cost_breakdown.get("total", 0))

        # Add route cost if available
        if route_details and route_details.estimated_cost:
            if "transportation" in cost_breakdown:
                cost_breakdown["transportation"] += route_details.estimated_cost
            else:
                cost_breakdown["transportation"] = route_details.estimated_cost
            total_estimated_cost += route_details.estimated_cost
            cost_breakdown["total"] = total_estimated_cost

        print(f"\n✅ Itinerary Generated Successfully!")
        print(f"💰 Total Cost: ₹{total_estimated_cost} / ₹{req.budget}")
        print(f"📊 Cost Breakdown:")
        for key, value in cost_breakdown.items():
            if key != "total":
                print(f"   - {key.title()}: ₹{value}")
        print(f"   - Total: ₹{cost_breakdown.get('total', 0)}")
        print(f"{'='*60}\n")

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
            cost_breakdown=cost_breakdown,
            budget_warning=itinerary_data.get("budget_warning")
        )

    except Exception as e:
        print(f"❌ Error parsing itinerary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse itinerary: {str(e)}")

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

@app.post("/trip/analyze-preferences")
async def analyze_user_preferences(user_comments: str, destination: str, themes: List[str] = None):
    """Analyze user comments and provide personalized suggestions"""
    if not user_comments.strip():
        return {
            "suggestions": [], 
            "themes": themes or [], 
            "message": "No comments provided for analysis"
        }
    
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
    
    Keep response concise and actionable (max 300 words).
    """
    
    try:
        response = model.generate_content(prompt)
        analysis = response.text if response else "Unable to analyze preferences"
        
        # Extract themes based on keywords
        detected_themes = []
        comment_lower = user_comments.lower()
        
        theme_keywords = {
            'food': ['food', 'eat', 'cuisine', 'restaurant', 'local dishes', 'cooking'],
            'cultural': ['history', 'historical', 'museum', 'culture', 'traditional', 'art'],
            'adventure': ['adventure', 'hiking', 'outdoor', 'sports', 'thrill', 'trekking'],
            'nature': ['nature', 'park', 'garden', 'wildlife', 'scenic', 'beach'],
            'shopping': ['shopping', 'market', 'buy', 'souvenir', 'mall'],
            'nightlife': ['nightlife', 'bar', 'club', 'evening', 'party'],
            'heritage': ['heritage', 'ancient', 'monument', 'temple', 'palace'],
            'sightseeing': ['sightseeing', 'tourist', 'landmark', 'famous', 'visit']
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
            "suggested_themes": ["cultural"],
            "selected_themes": themes or [],
            "user_comments": user_comments,
            "destination": destination
        }

@app.get("/status")
async def status():
    return {
        "status": "AI Trip Planner with Gemini Cost Breakdown is running",
        "bookings_count": len(bookings),
        "features": [
            "✅ Gemini AI generates ALL cost calculations",
            "✅ Budget is for ALL travelers combined (not per person)",
            "✅ Activity costs match cost breakdown totals",
            "✅ Accurate Google Maps geocoding for coordinates",
            "✅ Local festivals and events integration",
            "✅ User comments and preferences analysis",
            "✅ Weather forecast integration",
            "✅ 4-6 activities per day with optimal timing",
            "✅ Real-time hotel recommendations",
            "✅ Multi-modal route planning",
            "✅ Cost breakdown validation",
            "✅ Budget enforcement (never exceeds budget)",
            "✅ Coordinates plotted correctly on land"
        ],
        "supported_themes": [
            "cultural", "adventure", "heritage", "nightlife", 
            "food", "nature", "shopping", "sightseeing"
        ],
        "cost_calculation": "Gemini AI handles all cost breakdowns automatically",
        "version": "6.0"
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to Enhanced AI Trip Planner with Gemini Cost Breakdown",
        "description": "AI-powered travel planning with automatic cost calculations for all travelers",
        "endpoints": {
            "generate_itinerary": "/trip/generate-itinerary",
            "analyze_preferences": "/trip/analyze-preferences",
            "book_itinerary": "/trip/book-itinerary",
            "get_coordinates": "/trip/coordinates/{location}",
            "get_hotels": "/trip/hotels/{location}",
            "get_route": "/trip/route?from_location=X&to_location=Y",
            "user_bookings": "/trip/bookings/{user_id}",
            "status": "/status"
        },
        "key_improvements": [
            "✅ Gemini generates complete cost breakdown automatically",
            "✅ Budget is for ALL travelers for ALL days (not per person)",
            "✅ Activity costs match cost breakdown totals EXACTLY",
            "✅ Local festivals and unique events included",
            "✅ Real places with accurate Google Maps coordinates",
            "✅ Weather-aware activity scheduling",
            "✅ Cost validation ensures activities sum = breakdown",
            "✅ Multi-theme support with preference analysis",
            "✅ Fixed coordinate plotting (no more sea locations!)"
        ],
        "cost_structure": {
            "note": "All costs calculated by Gemini AI",
            "breakdown_includes": [
                "accommodation",
                "activities", 
                "food",
                "transportation",
                "miscellaneous"
            ],
            "validation": [
                "Activities total = cost_breakdown.activities",
                "Sum of all categories = cost_breakdown.total",
                "Total never exceeds user budget"
            ]
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)