# Cordoba - Terrain & Building Data Processor

A Python-based tool for processing and visualizing terrain elevation data, buildings, and road networks using Mapbox and OpenTopoData APIs.

## Features

- Geocode addresses to latitude/longitude coordinates
- Fetch and process satellite imagery from Mapbox
- Extract building footprints and road networks from map data
- Retrieve elevation data from OpenTopoData
- 3D terrain visualization with buildings and roads
- Interactive data manipulation and export

## Prerequisites

- Python 3.7+
- Mapbox API access token
- OpenTopoData API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Cordoba.git
cd Cordoba
```

2. Install dependencies:
```bash
pip install -r Docker/requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
OPENTOPO_API_KEY=your_opentopo_api_key_here
```

## API Keys

### Mapbox Access Token
Get your free Mapbox token at: https://account.mapbox.com/access-tokens/

### OpenTopoData API Key
Get your API key at: https://www.opentopodata.org/

## Usage

### Basic Usage (Command Line)

Process a specific address:
```bash
python shapetofunction.py --address "Your Address Here"
```

With custom API keys:
```bash
python shapetofunction.py \
  --address "Your Address Here" \
  --mapbox_token YOUR_TOKEN \
  --api_key_opentopo YOUR_KEY
```

### Flask Web Application

Run the web interface:
```bash
python apicalldisplay.py
```

Then navigate to `http://localhost:5000` in your browser.

### Docker Deployment

Build and run with Docker:
```bash
cd Docker
docker build -t cordoba-app .
docker run -p 80:80 cordoba-app
```

## Project Structure

```
Cordoba/
├── shapetofunction.py      # Main processing script
├── apicalldisplay.py        # Flask web application
├── plotting.py              # Visualization utilities
├── TileVector.py            # Vector tile processing
├── Docker/
│   ├── servercode.py        # Docker server implementation
│   └── requirements.txt     # Python dependencies
├── static/                  # Web frontend assets
│   ├── frontend.js
│   ├── css/
│   └── icons/
└── templates/               # HTML templates
    └── frontend.html
```

## Features in Detail

### Terrain Processing
- Fetches elevation data for specified geographic areas
- Creates interpolated elevation grids
- Supports custom zoom levels and area sizes

### Building Detection
- Identifies building footprints from satellite imagery
- Extracts polygon coordinates and centroids
- Classifies buildings by type and size

### Road Network Extraction
- Detects roads, highways, and freeways
- Skeletonizes road networks
- Simplifies paths using RDP algorithm
- Identifies intersections and endpoints

### 3D Visualization
- Interactive 3D terrain plots
- Overlays buildings and roads on elevation data
- Supports rotation, flipping, and transformation controls

## Output Format

Data is exported in JSON format with the following structure:
```json
{
  "elevation_data": {
    "elevation_values": [...],
    "x_values": [...],
    "y_values": [...],
    "lat_range": [...],
    "lon_range": [...],
    "min_elevation": 0,
    "max_elevation": 0
  },
  "buildings": [...],
  "roads": [...]
}
```

## Security Notes

**IMPORTANT:** Never commit API keys or sensitive data to version control!

- All API keys should be stored in `.env` file (which is gitignored)
- Remove any hardcoded addresses or personal location data before sharing
- Use environment variables for all sensitive configuration
- Run Flask in production mode for deployed instances

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your chosen license here - e.g., MIT, GPL, Apache 2.0]

## Acknowledgments

- [Mapbox](https://www.mapbox.com/) for map and geocoding services
- [OpenTopoData](https://www.opentopodata.org/) for elevation data
- Built with Python, Flask, NumPy, and scikit-image

## Disclaimer

This tool is for educational and research purposes. Ensure you comply with the terms of service for Mapbox and OpenTopoData APIs. Be mindful of API rate limits and usage quotas.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
