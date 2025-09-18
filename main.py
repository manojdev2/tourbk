from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
import json
import re

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

# --- Models ---
class TripRequest(BaseModel):
    location: str
    duration: int
    budget: int
    theme: str

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

class Itinerary(BaseModel):
    location: str
    duration: int
    budget: int
    theme: str
    days: List[ItineraryDay]
    total_estimated_cost: Optional[int] = None

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

    prompt = (
        f"Generate a detailed {user_inputs['duration']}-day travel itinerary for {user_inputs['location']} "
        f"focused on {user_inputs['theme']} theme, within a budget of INR {user_inputs['budget']}. "
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
        "- Include realistic costs in INR within the budget\n"
        f"- Focus on {user_inputs['theme']} theme\n"
        "- Use only these categories: sightseeing, food, adventure, cultural, shopping, nature, nightlife, heritage\n"
        "- Ensure valid JSON format with no comments, no trailing commas, no extra text\n"
        "- All property names must be in double quotes\n"
        "- All string values must be in double quotes\n"
        f"- Keep total under INR {user_inputs['budget']}\n"
        "- Include duration_hours as decimal numbers (e.g., 2.5)\n"
        "- Provide real coordinates for the specified destination"
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
    print(f"Creating fallback itinerary for {user_inputs['location']}")
    
    # Get coordinates for the location
    location_coords = get_location_coordinates_dict(user_inputs['location'])
    
    activities_per_day = max(1, user_inputs['budget'] // (user_inputs['duration'] * 1000))
    cost_per_activity = user_inputs['budget'] // (user_inputs['duration'] * activities_per_day)
    
    fallback_data = {
        "days": [],
        "total_estimated_cost": user_inputs['budget']
    }
    
    for day_num in range(1, user_inputs['duration'] + 1):
        day_activities = []
        
        for i in range(min(4, activities_per_day)):
            activity = {
                "name": f"Explore {user_inputs['location']} - Activity {i + 1}",
                "description": f"Discover the beauty and culture of {user_inputs['location']} with this {user_inputs['theme']} themed activity.",
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
    """Get coordinates for a location"""
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
    """Generate a detailed trip itinerary with activities, coordinates, and costs"""
    itinerary_data = generate_itinerary_with_genai(req.dict())

    if not itinerary_data or "days" not in itinerary_data:
        raise HTTPException(status_code=500, detail="Failed to generate itinerary")

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
        except (ValueError, TypeError):
            total_cost = calculate_total_cost(days)

        return Itinerary(
            location=req.location,
            duration=req.duration,
            budget=req.budget,
            theme=req.theme,
            days=days,
            total_estimated_cost=total_cost
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
        "booking_timestamp": "2025-09-18"  # In real app, use datetime.now()
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
    """Get basic coordinates for a location (placeholder for geocoding service)"""
    coords_dict = get_location_coordinates_dict(location)
    
    return LocationCoordinates(
        location=location,
        latitude=coords_dict["latitude"],
        longitude=coords_dict["longitude"]
    )

@app.get("/status")
async def status():
    """Get API status and statistics"""
    return {
        "status": "Generative AI Trip Planner with Google GenAI is running",
        "bookings_count": len(bookings),
        "features": [
            "AI-powered itinerary generation",
            "Individual day planning",
            "Activity coordinates for mapping",
            "Cost estimation",
            "Booking system",
            "Theme-based recommendations",
            "Robust JSON parsing",
            "Fallback itinerary generation"
        ],
        "version": "2.1"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the AI Trip Planner API",
        "description": "Generate personalized travel itineraries with AI",
        "endpoints": {
            "generate_itinerary": "/trip/generate-itinerary",
            "book_itinerary": "/trip/book-itinerary", 
            "get_coordinates": "/trip/coordinates/{location}",
            "user_bookings": "/trip/bookings/{user_id}",
            "status": "/status"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)