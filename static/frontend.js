document.addEventListener("DOMContentLoaded", function () {
  function fetchDataAndPlot(url, body, handleData) {
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: body
    })
      .then(response => {
        console.log(`Response from ${url}:`, response);
        return response.json();
      })
      .then(data => {
        console.log(`Received full JSON data from ${url}:`, data);
        handleData(data);
      })
      .catch(error => console.error(`Error fetching data from ${url}:`, error));
  }

  async function handleAssemblyData(data) {
    const voxelPlotElement = document.getElementById('voxelplot');

    if (!voxelPlotElement) {
      console.error('voxelplot element not found');
      return;
    }

    if (!data.terrain_data) {
      console.error('terrain_data is missing or invalid');
      return;
    }

    console.log('Terrain data received:', data.terrain_data);
    voxelPlotElement.innerHTML = ''; // Clear previous plot

    const layout = getLayoutSettings(data.terrain_data);
    const traces = [];
  
    const objectTraces = plotObjects(data.object_data, data.terrain_data);
    traces.push(...objectTraces);
  
    // Plot buildings with adjusted opacity
    const buildingTraces = await plot3DModels(data.object_data, data.terrain_data, 0.7);
    traces.push(...buildingTraces);
  
    Plotly.newPlot(voxelPlotElement, traces, layout);
  }

  function getLayoutSettings(terrainData) {
    const xValues = terrainData.points.map(point => point[0]);
    const yValues = terrainData.points.map(point => point[1]);
    const zValues = terrainData.points.map(point => point[2]);
    const minZ = Math.min(...zValues);
    const maxZ = minZ + 100; // Set the maximum z-value to the mi
    return {
      scene: {
        xaxis: {
          range: [Math.min(...xValues), Math.max(...xValues)]
        },
        yaxis: {
          range: [Math.min(...yValues), Math.max(...yValues)]
        },
        zaxis: {
          range: [minZ, maxZ]
        },
        aspectmode: 'manual',
        aspectratio: {
          x: 1,
          y: 1,
          z: 0.5
        }
      },
      // Include other layout settings if necessary
    };
  }
  function plotObjects(objectData, terrainData) {
    const traces = [];
  
    // Create a separate trace for terrain points
    const terrainTrace = {
      x: terrainData.points.map(point => point[0]),
      y: terrainData.points.map(point => point[1]),
      z: terrainData.points.map(point => point[2]),
      mode: 'markers',
      marker: {
        size: 3,
        color: 'green'
      },
      name: 'Terrain',
      type: 'scatter3d'
    };
    traces.push(terrainTrace);
  
    for (const [type, objects] of Object.entries(objectData)) {
      if (type !== 'terrain') {
        const objectTrace = {
          x: [],
          y: [],
          z: [],
          mode: 'markers',
          marker: {
            size: 3,
            color: getColorForLabel(type)
          },
          name: type,
          type: 'scatter3d'
        };
  
        for (const object of Object.values(objects)) {
          if (object.points) {
            object.points.forEach(([x, y, z]) => {
              objectTrace.x.push(x);
              objectTrace.y.push(y);
              objectTrace.z.push(z);
            });
          }
        }
  
        traces.push(objectTrace);
      }
    }
  
    return traces;
  }
function getZPosition(point, terrainData) {
  const { x, y } = point;
  const nearestPoint = terrainData.points.reduce((nearest, curr) => {
    const dx = curr[0] - x;
    const dy = curr[1] - y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    return distance < nearest.distance ? { point: curr, distance } : nearest;
  }, { point: null, distance: Infinity });

  return nearestPoint.point[2];
}
function createHouseTrace(houseObjUrl, x, y, z, opacity = 1) {
  return new Promise((resolve, reject) => {
    Plotly.d3.text(houseObjUrl, function (text) {
      if (text) {
        const objLines = text.split('\n');
        const vertexData = objLines.filter(line => line.startsWith('v ')).map(line => line.slice(2).split(' ').map(Number));
        const faceData = objLines.filter(line => line.startsWith('f ')).map(line => line.slice(2).split(' ').map(v => parseInt(v.split('/')[0], 10) - 1));

        // Adjust scaling factors
        const scaleX = 0.1; // Example adjustment
        const scaleY = 0.1; // Example adjustment
        const scaleZ = 0.1; // Example adjustment

        const scaledVertexData = vertexData.map(vertex => [
          vertex[0] * scaleX,
          vertex[1] * scaleY,
          vertex[2] * scaleZ
        ]);

        const houseTrace = {
          type: 'mesh3d',
          x: scaledVertexData.map(v => v[0] + x),
          y: scaledVertexData.map(v => v[1] + y),
          z: scaledVertexData.map(v => v[2] + z),
          i: faceData.map(f => f[0]),
          j: faceData.map(f => f[1]),
          k: faceData.map(f => f[2]),
          color: 'rgb(255, 0, 0)',
          opacity: opacity,
          flatshading: false
        };

        resolve(houseTrace);
      } else {
        reject(new Error('Failed to load house.obj file'));
      }
    });
  });
}

  async function plot3DModels(objectData, terrainData, opacity = 1) {
    const traces = [];
  
    for (const [type, objects] of Object.entries(objectData)) {
      if (type === 'houses') {
        for (const object of Object.values(objects)) {
          const centerPoint = getObjectCenter(object.points);
          const { x, y, z } = centerPoint;
          const houseTrace = await createHouseTrace('/static/house.obj', x, y, z, opacity);
          traces.push(houseTrace);
        }
      }
    }
  
    return traces;
  }
  function getObjectCenter(points) {
    // Logic to find the center point of an object
    let xSum = 0, ySum = 0, zSum = 0;
    points.forEach(point => {
      xSum += point[0];
      ySum += point[1];
      zSum += point[2];
    });
    return { x: xSum / points.length, y: ySum / points.length, z: zSum / points.length };
  }

  function getColorForLabel(label) {
    const colorMap = {
      'houses': 'tomato',
      'roads': 'steelblue',
      'terrain': 'green'
    };
    return colorMap[label] || 'purple';
  }

  document.getElementById('address-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const address = document.getElementById('address').value;
    fetchDataAndPlot('/layerassembly', `address=${encodeURIComponent(address)}`, handleAssemblyData);
  });

  $('#map-icon').click(function () {
    $('#address-form').submit();
  });

  const sendBtn = document.getElementById('sendBtn');
  const messageInput = document.getElementById('messageInput');
  const chatArea = document.getElementById('chat');

  sendBtn.addEventListener('click', function () {
    console.log("button clicked");
    const message = messageInput.value.trim();
    if (message) {
      appendMessageToChat(message, true);
      appendMessageToChat("Making Request");
      fetchDataAndPlot('/generate-voxel', `text=${encodeURIComponent(message)}`, handleAssemblyData);
      messageInput.value = '';
    } else {
      console.error('Message input is empty.');
    }
  });

  const sidebar = document.querySelector('.sidebar');
  const content = document.querySelector('.content');
  const collapseBtn = document.querySelector('.collapse-btn');
  const expandBtn = document.querySelector('.expand-btn');

  collapseBtn.addEventListener('click', function () {
    sidebar.classList.add('collapsed');
    content.classList.add('expanded');
  });

  expandBtn.addEventListener('click', function () {
    sidebar.classList.remove('collapsed');
    content.classList.remove('expanded');
  });

  function appendMessageToChat(message, isUserMessage = false) {
    const chatMessage = document.createElement('div');
    chatMessage.textContent = message;
    chatMessage.classList.add('chat-message');

    if (isUserMessage) {
      chatMessage.classList.add('user-message');
    } else {
      chatMessage.classList.add('bot-message');
    }

    chatArea.appendChild(chatMessage);
  }
});