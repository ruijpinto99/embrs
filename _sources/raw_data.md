# Generating Raw Data/Input Files
Before using the provided map generator tool and creating a EMBRS map, there is some raw data that must be gathered.

- Fuel map (.tif)
- Elevation map (.tif)

## Fuel and Elevation Data
EMBRS is designed to be used with data acquired from the LANDFIRE database which is a U.S. government program that provides
geospatial wildland fire data. This gives users the ability to run fire scenarios using realistic landscapes. 

```{note}
Fuel and elevation data is not necessary to run a sim. When generating a map there are options to create uniform fuel and/or
elevation maps, but LANDFIRE data is the only supported method of creating heterogeneous fuel and elevation maps.
```

### **Step 1: Open LANDFIRE Data Map**

- Navigate to [landfire.gov](https://www.landfire.gov)
- Click on the 'Get Data' icon


```{figure} /images/lf_tutorial_get_data_link.png
---
scale: 75%
---
```

- Under 'Map Viewer' click on 'LF Map Viewer'

```{figure} /images/lf_tutorial_map_viewer_link.png
---
scale: 75%
---
```

- You should now see an interactive map of the United States with some data overlay.

```{figure} /images/lf_tutorial_map_view.png
---
scale: 75%
---
```

### **Step 2: Select Region of Interest**

- Pan and zoom to the region you are interested in.
- Once the region is within your view, click data download icon in the top toolbar.

```{figure} /images/lf_tutorial_data_download_tool.png
---
scale: 60%
---
```

- Your cursor should now be a blue circle. Ensure the 'Method' option on the right pane
is set to 'Rectangle'. Now, you can click to draw a bounding rectangle that determines the region of interest.

```{figure} /images/lf_tutorial_selected_region.png
---
scale: 75%
---
```

- The drawing tool will tell you the area you currently have selected in square miles.


### **Step 3: Select Data Products**
- Once you have drawn the bounding rectangle, you need to select the correct data to download. 
- In the 'tools' pane on the right, expand the 'LF 2022 (LF_230)' folder, within that folder expand 'Fuel' and then 'Surface and Canopy'.
- Check the box next to the option for 'us_230 13 Fire Behavior Fuel Models-Anderson' this is the fuel model used by EMBRS.

```{figure} /images/lf_tutorial_data_product_fuel.png
---
scale: 100%
---
```


- Next expand the folder named 'Topographic' and check the box next to 'us_220 Elevation' this is the elevation map data.

```{figure} /images/lf_tutorial_data_product_elev.png
---
scale: 100%
---
```

- Make sure these are the only boxes checked in the right pane.

```{note} 
If you would like to inspect the data visually before downloading you can select the same options in the left pane to get
a visual representation of it.
```

### **Step 4: Request Data**
- With both the fuel and elevation data products selected, enter your email in the 'Email' entry box.
- Click download to receive the data via email, it can sometimes take several minutes to receive the data.

```{figure} /images/lf_tutorial_email_entry.png
---
scale: 100%
---
```


### **Step 5: Save Data**
- The data will be sent to you as an email with a link to download a zip file.

```{figure} /images/lf_tutorial_email.png
---
scale: 75%
---
```


- You can change the name and location of the parent folder after unzipping, but keep the two folders within it
exactly as they are (see below image).

```{figure} /images/lf_tutorial_file_struct.png
---
scale: 75%
---
```

- You are now set to import this data to create an EMBRS map.