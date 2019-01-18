import numpy as np

#=============================================================================================
# Color functions
#=============================================================================================

#Returns the hex string corresponding to the passed rgb values
def rgb(r,g,b):
    r = int(r)
    g = int(g)
    b = int(b)
    return '#%02x%02x%02x' % (r, g, b)
    
#Computes a hex value matching up with the current position relative to the range of colors.
#position = current position within the range of colors (e.g., 1)
#rng = range of colors (e.g. 5, so 1/5 would be 20% of the range)
#col1 = Starting RGB color for the range (e.g. [0,255,255])
#col2 = Ending RGB color for the range (e.g. [0,0,255])
def getColor(position,rng,col1,col2):
    
    #Retrieve r,g,b values from tuple
    r1,g1,b1 = col1
    r2,g2,b2 = col2

    #Get difference in each r,g,b value between start & end 
    rdif = float(r2 - r1)
    gdif = float(g2 - g1)
    bdif = float(b2 - b1)
    
    #Calculate r,g,b values for the specified position within the range
    r3 = r2 + (-1.0 * position * (rdif / float(rng)))
    g3 = g2 + (-1.0 * position * (gdif / float(rng)))
    b3 = b2 + (-1.0 * position * (bdif / float(rng)))

    #Return in hex string format
    return rgb(r3,g3,b3)
    
#Returns a list of colors matching up with the range(s) provided
def list_by_range(*args):
    
    #Initialize an empty color list
    colors = []
    
    #Loop through the passed lists, following a format of ['#0000FF','#00FFFF',5]
    #where [0] = start color, [1] = end color, [2] = number of colors in range
    for arg in args:
        
        #Retrieve arguments
        start_hex = arg[1].lstrip('#')
        end_hex = arg[0].lstrip('#')
        nrange_color = arg[2]-1
        
        #Calculate start and end RGB tuples
        start_rgb = tuple(int(start_hex[i:i+2], 16) for i in (0, 2 ,4))
        end_rgb = tuple(int(end_hex[i:i+2], 16) for i in (0, 2 ,4))
            
        #Loop through the number of colors to add into the list
        for x in range(0,nrange_color+1):

            #Get hex value for the color at this point in the range
            hex_val = getColor(x,nrange_color,start_rgb,end_rgb)
            
            #Append to list if this is different than the last color
            if len(colors) == 0 or colors[-1] != hex_val: colors.append(hex_val)
            
    #Return the list of colors
    return colors

#Returns a list of colors matching up with the range of numerical values provided
def list_by_values(*args):
    
    #Initialize an empty color list
    colors = []
    
    #Loop through the passed lists
    #The format for each argument is: [['#00FFFF',25.0],['#0000FF',29.0],1.0]
    #['#00FFFF',25.0] = [start hex value, start value]
    #['#0000FF',29.0] = [end hex value, end value]
    #1.0 = interval of requested range between start & end values
    for arg in args:
        
        #Retrieve arguments
        start_hex = arg[1][0].lstrip('#')
        end_hex = arg[0][0].lstrip('#')
        start_value = arg[0][1]
        end_value = arg[1][1]
        interval = arg[2]
        
        #Calculate start and end RGB tuples
        start_rgb = tuple(int(start_hex[i:i+2], 16) for i in (0, 2 ,4))
        end_rgb = tuple(int(end_hex[i:i+2], 16) for i in (0, 2 ,4))
            
        #Loop through the number of colors to add into the list
        start_loop = start_value
        end_loop = end_value + interval if arg == args[-1] else end_value
        for x in np.arange(start_loop,end_loop,interval):

            #Get hex value for the color at this point in the range
            nrange_color = (end_value - start_value)
            idx = x - start_value
            hex_val = getColor(idx,nrange_color,start_rgb,end_rgb)
            
            #Append to list if this is different than the last color
            if len(colors) == 0 or colors[-1] != hex_val: colors.append(hex_val)
            
    #Return the list of colors
    return colors
    
#=============================================================================================
# Sample Usage
#=============================================================================================

#This is the call to the list_by_range function. Input parameters are as many ranges
#as desired. In this example, there are 2 ranges, one starting from #00FFFF and
#going to #0000FF, with 5 values in the range (where value 1 is the starting color
#and value 5 is the ending color).
colors = list_by_range(['#00FFFF','#0000FF',5],['#008A05','#00FF09',10])
print(colors)

#This is the call to the list_by_values function. Input parameters are as many ranges
#as desired. In this example, there are 3 ranges, one starting from #00FFFF for a value
#of 25 and going to #0000FF for a value of 29.
colors = list_by_values([['#00FFFF',25.0],['#0000FF',29.0],1.0],
                        [['#0000FF',29.0],['#0000AA',32.0],1.0],
                        [['#0000AA',32.0],['#FF00FF',38.0],1.0])
print(colors)