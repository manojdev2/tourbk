from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
import json
import re
import requests
from datetime import datetime

import google.generativeai as genai

# Google GenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Generative AI Trip Planner with Google GenAI SDK")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Maps API Key - Replace with your actual API key
GOOGLE_MAPS_API_KEY = "AIzaSyDAUhNkL--7MVKHtlFuR3acwa7ED-cIoAU"

# --- Models ---
class TripRequest(BaseModel):
    location: str  # Kept for backward compatibility
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

class ItineraryDay(BaseModel):
    day: int
    activities: List[Activity]
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
    """Get coordinates using Google Geocoding API"""
    url = f"https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': location,
        'key': GOOGLE_MAPS_API_KEY
    }
    
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
    
    # Fallback to local dictionary
    return get_location_coordinates_dict(location)

def find_hotels_near_location(location: str, radius: int = 5000) -> List[Hotel]:
    """Find hotels near the destination using Google Places API"""
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
            for place in data.get('results', [])[:10]:  # Limit to 10 hotels
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
    """Get route details using Google Directions API"""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Map preferred_transport to Google Maps travel modes
    mode_mapping = {
        "driving": "driving",
        "car": "driving",
        "walking": "walking",
        "transit": "transit",
        "public_transport": "transit",
        "bicycling": "bicycling",
        "bike": "bicycling"
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
            
            # Estimate travel cost based on mode and distance
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
                } for step in leg.get('steps', [])[:5]]  # Limit to first 5 steps
            )
    
    except Exception as e:
        print(f"Error getting route details: {e}")
    
    return None

def calculate_travel_cost(distance_meters: int, travel_mode: str) -> int:
    """Calculate estimated travel cost based on distance and mode"""
    distance_km = distance_meters / 1000
    
    if travel_mode == "driving":
        # Estimate: ₹8 per km (fuel + toll + wear)
        return int(distance_km * 8)
    elif travel_mode == "transit":
        # Estimate: ₹2 per km for public transport
        return int(distance_km * 2)
    elif travel_mode == "bicycling":
        # Minimal cost for bike rental
        return min(500, int(distance_km * 5))
    elif travel_mode == "walking":
        # No cost for walking
        return 0
    
    return int(distance_km * 10)  # Default estimate

# --- GenAI client ---
def get_genai_model():
    genai.configure(api_key='AIzaSyCn43FyMu0k4TpBrrXVo1KNRtPR1JuUoF4')
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model

def clean_json_string(json_string: str) -> str:
    """Clean and fix common JSON formatting issues"""
    # Remove markdown code blocks
    if "```json" in json_string:
        json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
    
    # Remove any leading/trailing whitespace
    json_string = json_string.strip()
    
    # Remove comments (// comments)
    json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
    
    # Remove /* */ style comments
    json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
    
    # Fix common trailing comma issues
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)
    
    # Fix missing quotes around property names
    json_string = re.sub(r'(\w+)(\s*:)', r'"\1"\2', json_string)
    
    # Fix already quoted property names (avoid double quotes)
    json_string = re.sub(r'""(\w+)""(\s*:)', r'"\1"\2', json_string)
    
    return json_string

def validate_json_structure(data: dict) -> bool:
    """Validate that the JSON has the expected structure"""
    if not isinstance(data, dict):
        return False
    
    if "days" not in data:
        return False
    
    if not isinstance(data["days"], list):
        return False
    
    for day in data["days"]:
        if not isinstance(day, dict):
            return False
        if "day" not in day or "activities" not in day:
            return False
        if not isinstance(day["activities"], list):
            return False
        
        for activity in day["activities"]:
            if not isinstance(activity, dict):
                return False
            if "name" not in activity:
                return False
    
    return True

# --- Generate Itinerary with JSON Response ---
def generate_itinerary_with_genai(user_inputs: dict) -> dict:
    model = get_genai_model()
    
    # Use to_location if available, otherwise fall back to location
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    
    # Build enhanced prompt with new parameters
    traveler_info = f"for {user_inputs.get('traveler_count', 1)} traveler(s)" if user_inputs.get('traveler_count') else ""
    start_date_info = f"starting from {user_inputs.get('start_date')}" if user_inputs.get('start_date') else ""
    transport_info = f"preferring {user_inputs.get('preferred_transport', 'any')} transport" if user_inputs.get('preferred_transport') else ""

    prompt = (
        f"Generate a detailed {user_inputs['duration']}-day travel itinerary for {destination} "
        f"focused on {user_inputs['theme']} theme, within a budget of INR {user_inputs['budget']} "
        f"{traveler_info} {start_date_info} {transport_info}. "
        f"Return ONLY a valid JSON object with no additional text, comments, or markdown formatting. "
        f"The JSON structure should be:\n\n"
        "{\n"
        '  "days": [\n'
        "    {\n"
        '      "day": 1,\n'
        '      "activities": [\n'
        "        {\n"
        '          "name": "Activity Name",\n'
        '          "description": "Detailed description of the activity (2-3 sentences)",\n'
        '          "latitude": 12.9716,\n'
        '          "longitude": 77.5946,\n'
        '          "estimated_cost": 500,\n'
        '          "duration_hours": 2.5,\n'
        '          "category": "sightseeing"\n'
        "        }\n"
        "      ],\n"
        '      "total_day_cost": 2000\n'
        "    }\n"
        "  ],\n"
        '  "total_estimated_cost": 8000\n'
        "}\n\n"
        "CRITICAL REQUIREMENTS:\n"
        f"- Create exactly {user_inputs['duration']} days\n"
        "- Include 4-6 activities per day\n"
        "- Provide accurate latitude and longitude coordinates for each activity\n"
        f"- Include realistic costs in INR within the budget (consider {user_inputs.get('traveler_count', 1)} travelers)\n"
        f"- Focus on {user_inputs['theme']} theme\n"
        "- Use only these categories: sightseeing, food, adventure, cultural, shopping, nature, nightlife, heritage\n"
        "- Ensure valid JSON format with no comments, no trailing commas, no extra text\n"
        "- All property names must be in double quotes\n"
        "- All string values must be in double quotes\n"
        f"- Keep total under INR {user_inputs['budget']}\n"
        "- Include duration_hours as decimal numbers (e.g., 2.5)\n"
        "- Provide real coordinates for the specified destination"
        f"- Consider the preferred transport mode: {user_inputs.get('preferred_transport', 'any')}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent formatting
                    "top_p": 0.8,
                    "candidate_count": 1,
                },
            )

            output_text = response.text or ""
            
            # Clean the JSON string
            cleaned_json = clean_json_string(output_text)
            
            print(f"Attempt {attempt + 1} - Cleaned JSON: {cleaned_json[:500]}...")  # Log first 500 chars
            
            # Try to parse JSON
            try:
                parsed_data = json.loads(cleaned_json)
                
                # Validate structure
                if validate_json_structure(parsed_data):
                    print(f"Successfully parsed JSON on attempt {attempt + 1}")
                    return parsed_data
                else:
                    print(f"Invalid JSON structure on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return create_fallback_itinerary(user_inputs)
                    
            except json.JSONDecodeError as json_error:
                print(f"JSON Parse Error (attempt {attempt + 1}): {json_error}")
                print(f"Problematic JSON: {cleaned_json}")
                
                if attempt == max_retries - 1:
                    return create_fallback_itinerary(user_inputs)

        except Exception as e:
            print(f"Error calling GenAI SDK (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return create_fallback_itinerary(user_inputs)

    return {}

def create_fallback_itinerary(user_inputs: dict) -> dict:
    """Create a basic fallback itinerary when GenAI fails"""
    destination = user_inputs.get('to_location') or user_inputs.get('location')
    print(f"Creating fallback itinerary for {destination}")
    
    # Get coordinates for the location
    location_coords = get_location_coordinates_dict(destination)
    
    traveler_count = user_inputs.get('traveler_count', 1)
    activities_per_day = max(1, user_inputs['budget'] // (user_inputs['duration'] * 1000 * traveler_count))
    cost_per_activity = user_inputs['budget'] // (user_inputs['duration'] * activities_per_day)
    
    fallback_data = {
        "days": [],
        "total_estimated_cost": user_inputs['budget']
    }
    
    for day_num in range(1, user_inputs['duration'] + 1):
        day_activities = []
        
        for i in range(min(4, activities_per_day)):
            activity = {
                "name": f"Explore {destination} - Activity {i + 1}",
                "description": f"Discover the beauty and culture of {destination} with this {user_inputs['theme']} themed activity.",
                "latitude": location_coords['latitude'] + (i * 0.01),
                "longitude": location_coords['longitude'] + (i * 0.01),
                "estimated_cost": cost_per_activity,
                "duration_hours": 2.0,
                "category": user_inputs['theme'] if user_inputs['theme'] in ['sightseeing', 'food', 'adventure', 'cultural', 'shopping', 'nature', 'nightlife', 'heritage'] else 'sightseeing'
            }
            day_activities.append(activity)
        
        fallback_data["days"].append({
            "day": day_num,
            "activities": day_activities,
            "total_day_cost": cost_per_activity * len(day_activities)
        })
    
    return fallback_data

def get_location_coordinates_dict(location: str) -> dict:
    """Get coordinates for a location (fallback dictionary)"""
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
    if location_lower in location_coords:
        return location_coords[location_lower]
    
    # Try partial matching
    for key in location_coords:
        if key in location_lower or location_lower in key:
            return location_coords[key]
    
    # Default fallback
    return {"latitude": 12.9716, "longitude": 77.5946}

# --- Helper Functions ---
def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate if coordinates are within valid ranges"""
    return -90 <= latitude <= 90 and -180 <= longitude <= 180

def calculate_total_cost(days: List[ItineraryDay]) -> int:
    """Calculate total estimated cost from all days"""
    total = 0
    for day in days:
        if day.total_day_cost:
            total += day.total_day_cost
        else:
            # Fallback: sum individual activity costs
            day_total = sum(activity.estimated_cost or 0 for activity in day.activities)
            total += day_total
    return total

# --- Dummy Payment ---
def process_payment(payment_token: str) -> bool:
    return True  # extend with real payment later

# --- API Routes ---
@app.post("/trip/generate-itinerary", response_model=Itinerary)
async def generate_itinerary(req: TripRequest):
    """Generate a detailed trip itinerary with activities, coordinates, costs, hotels, and route details"""
    
    # Generate itinerary data
    itinerary_data = generate_itinerary_with_genai(req.dict())

    if not itinerary_data or "days" not in itinerary_data:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary")

    # Get hotels near destination
    hotels = []
    if req.to_location:
        hotels = find_hotels_near_location(req.to_location)

    # Get route details if both locations are provided
    route_details = None
    if req.from_location and req.to_location:
        route_details = get_route_details(
            req.from_location, 
            req.to_location, 
            req.preferred_transport or "driving"
        )

    try:
        days = []
        for day_data in itinerary_data.get("days", []):
            activities = []
            
            for activity_data in day_data.get("activities", []):
                # Validate coordinates if provided
                lat = activity_data.get("latitude")
                lng = activity_data.get("longitude")
                
                if lat is not None and lng is not None:
                    try:
                        lat = float(lat)
                        lng = float(lng)
                        if not validate_coordinates(lat, lng):
                            print(f"Invalid coordinates for {activity_data.get('name')}: {lat}, {lng}")
                            lat, lng = None, None
                    except (ValueError, TypeError):
                        print(f"Invalid coordinate format for {activity_data.get('name')}: {lat}, {lng}")
                        lat, lng = None, None
                
                # Validate cost and duration
                cost = activity_data.get("estimated_cost", 0)
                duration = activity_data.get("duration_hours")
                
                try:
                    cost = int(cost) if cost is not None else 0
                    # Adjust cost for multiple travelers
                    if req.traveler_count and req.traveler_count > 1:
                        cost = int(cost * req.traveler_count)
                except (ValueError, TypeError):
                    cost = 0
                
                try:
                    duration = float(duration) if duration is not None else None
                except (ValueError, TypeError):
                    duration = None
                
                activity = Activity(
                    name=activity_data.get("name", "Unknown Activity"),
                    description=activity_data.get("description", "No description available"),
                    latitude=lat,
                    longitude=lng,
                    estimated_cost=cost,
                    duration_hours=duration,
                    category=activity_data.get("category", "general")
                )
                activities.append(activity)
            
            day_cost = day_data.get("total_day_cost")
            try:
                day_cost = int(day_cost) if day_cost is not None else None
                # Adjust day cost for multiple travelers
                if day_cost and req.traveler_count and req.traveler_count > 1:
                    day_cost = int(day_cost * req.traveler_count)
            except (ValueError, TypeError):
                day_cost = None
            
            day = ItineraryDay(
                day=day_data.get("day", 0),
                activities=activities,
                total_day_cost=day_cost
            )
            days.append(day)

        # Calculate total cost if not provided
        total_cost = itinerary_data.get("total_estimated_cost")
        try:
            total_cost = int(total_cost) if total_cost is not None else calculate_total_cost(days)
            # Add route cost if available
            if route_details and route_details.estimated_cost:
                total_cost += route_details.estimated_cost
        except (ValueError, TypeError):
            total_cost = calculate_total_cost(days)

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
        print(f"Raw data: {itinerary_data}")
        raise HTTPException(status_code=500, detail=f"Failed to parse itinerary data: {str(e)}")

@app.post("/trip/book-itinerary")
async def book_itinerary(req: BookingRequest):
    """Book a generated itinerary"""
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
    """Get all bookings for a specific user"""
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
    """Get coordinates for a location using Google Geocoding API"""
    coords_dict = get_location_coordinates_from_google(location)
    
    return LocationCoordinates(
        location=location,
        latitude=coords_dict["latitude"],
        longitude=coords_dict["longitude"]
    )

@app.get("/trip/hotels/{location}")
async def get_hotels(location: str, radius: int = 5000):
    """Get hotels near a location"""
    hotels = find_hotels_near_location(location, radius)
    return {
        "location": location,
        "hotels": hotels,
        "count": len(hotels)
    }

@app.get("/trip/route")
async def get_route(from_location: str, to_location: str, travel_mode: str = "driving"):
    """Get route details between two locations"""
    route = get_route_details(from_location, to_location, travel_mode)
    if route:
        return route
    else:
        raise HTTPException(status_code=404, detail="Route not found")

@app.get("/status")
async def status():
    """Get API status and statistics"""
    return {
        "status": "Generative AI Trip Planner with Google GenAI and Maps API is running",
        "bookings_count": len(bookings),
        "features": [
            "AI-powered itinerary generation",
            "Individual day planning",
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
        "version": "3.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Enhanced AI Trip Planner API",
        "description": "Generate personalized travel itineraries with AI, hotel recommendations, and route planning",
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
            "Hotel recommendations from Google Places",
            "Route planning with multiple transport modes",
            "Multi-traveler cost calculation",
            "Enhanced location geocoding"
        ],
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)