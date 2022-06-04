#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from pandas.io.formats import style
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
import random
import os
import base64


# In[2]:


def offencenum(entered_year):
    offencedf = pd.read_csv('Final_Cleaned_Offence_3302021.csv')
    offencedf.drop(['Unnamed: 0'],axis=1,inplace=True)
    offencedf. astype(str)
    if(entered_year=='2018-2021'):
        data = offencedf.copy()
    else:
        data = offencedf[offencedf['year']==int(entered_year)]
        
    numoffence = len(data)
    offencegrp = data.groupby('Offence_name').count()['ID']
    offenceid = offencegrp.idxmax()
    

    
    
    return ['{:,}'.format(numoffence),offenceid]


# In[3]:


def timedata(entered_year):
    offencedf = pd.read_csv('Final_Cleaned_Offence_3302021.csv')
    offencedf.drop(['Unnamed: 0'],axis=1,inplace=True)
    offencedf. astype(str)
    
    grptime=offencedf.groupby(['year','Season','months','weekday','Hour']).count()[['Offence_name']]
    grptime.reset_index(inplace=True)
    
    grptime.rename(columns={'Offence_name':'Num_Offences'},inplace=True)
    if(entered_year=='2018-2021'):
        data = grptime.copy()
        dfs=data[['year','Season','Num_Offences']].copy()
        
        dfm=data[['year','months','Num_Offences']].copy()
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        dfm.months=pd.Categorical(dfm.months,categories=month,ordered=True)
        dfm.sort_values(by=['months','year'],inplace=True)
        
        dfd=data[['year','weekday','Num_Offences']].copy()
        cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dfd.weekday = pd.Categorical(dfd.weekday,categories=cats, ordered=True)
        dfd.reset_index(inplace=True)
        
        
    else:
        data = grptime[grptime['year']==int(entered_year)]
        dfs=data[['year','Season','Num_Offences']].copy()
        dfs=dfs.groupby('Season').sum()[['Num_Offences']]
        dfs.reset_index(inplace=True)
        
        dfm=data[['year','months','Num_Offences']].copy() 
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        dfm.months=pd.Categorical(dfm.months,categories=month,ordered=True)
        dfm.reset_index(inplace=True)
        dfm=dfm.groupby('months').sum()[['Num_Offences']]
        dfm.reset_index(inplace=True)
        
        dfd=data[['year','weekday','Num_Offences']].copy()
        cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dfd.weekday = pd.Categorical(dfd.weekday,categories=cats, ordered=True)
        dfd.reset_index(inplace=True)
        dfd=dfd.groupby('weekday').sum()[['Num_Offences']]
        dfd.reset_index(inplace=True)
    
    dft=data[['year','Hour','Num_Offences']].copy()
    dft=dft.groupby(['Hour','year']).sum()[['Num_Offences']]
    dft.reset_index(inplace=True)

    return [dfs,dfm,dfd,dft]


# In[4]:


def drawfigtime(entered_year):
    [d1,d2,d3,d4]=timedata(entered_year)
    if(entered_year=='2018-2021'):
        fig1=px.treemap(d1,path=['year','Season'],values='Num_Offences',title='Total number of offences by Season (2018-2021)')
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig2=px.treemap(d2,path=['months','year'],values='Num_Offences',title='Total number of offences by Month (2018-2021)')
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig3= px.sunburst(d3,path=['year','weekday'],values='Num_Offences',color='weekday',title='Number of offences by Day (2018-2021)')
        fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig4 = px.line(d4, x='Hour', y='Num_Offences',color='year',title='Total Number of Offences on Time basis (2018-2021)',
             labels = {"Num_Offences" : 'Total Number of Offences', "Hour" : 'Time'},markers=True)
        fig4.update_layout(autosize = False,
                 width = 800,
                 height=500,
                 margin = dict(t=30, l=25, r=25, b=50),
                 xaxis = dict(tickvals = list(np.arange(0,25)),
                 ticktext = ['00-01','01-02','02-03','03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-00','No Records'],
                 title_font_size = 14),
                 legend=dict(yanchor="top", y=0.99, xanchor="right",x=0.99))
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
    else:
        fig1 = px.bar(d1, x='Season', y='Num_Offences',color='Season',title='Total Number of Offences over the Season (2018-2021)',
             labels = {"Num_Offences" : 'Total Number of Offences', "year" : 'Season'})
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig2 = px.bar(d2, x='months', y='Num_Offences',color='months',color_discrete_sequence=['black','blue','red','pink','green','Gray','orange','purple','brown','cyan','olive','lime'],
             title='Total Number of Offences over the Months (2018-2021)',
             labels = {"Num_Offences" : 'Total Number of Offences', "months" : 'Months'},)
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig3=px.pie(d3,values='Num_Offences',names='weekday',title='Percentage of offences over the days (2018-2021)')
        fig3.update_traces(textposition='outside',textinfo='percent + label')
        fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        
        fig4 = px.line(d4, x='Hour', y='Num_Offences',color='year',title='Total Number of Offences on Time (2018-2021)',
             labels = {"Num_Offences" : 'Total Number of Offences', "Hour" : 'Time'},markers=True)
        fig4.update_layout(autosize = False,
                 width = 800,
                 height=500,
                 margin = dict(t=30, l=25, r=25, b=50),
                 xaxis = dict(tickvals = list(np.arange(0,25)),
                 ticktext = ['00-01','01-02','02-03','03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-00','No Records'],
                 title_font_size = 14),
                 legend=dict(yanchor="top", y=0.99, xanchor="right",x=0.99))
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
    return[fig1,fig2,fig3,fig4]


# In[5]:


def geodata(entered_year):
    df = pd.read_csv('rinchen6.csv')
    df.drop(['Unnamed: 0','Driving_Region'],axis=1,inplace=True)
    
    if(entered_year=='2018-2021'):
        df = df.copy()
        
        
    else:
        df = df[df['year']==int(entered_year)].copy()
    g2 = df.groupby(['region_name','Reg_Region']).count()[['ID']]
    g2.rename(columns={'ID':'Num_Offences'},inplace=True)
    g2.reset_index(inplace=True)
    
    df1=df.groupby(['base_office_name','Place_Of_Inspection']).count()[['ID']].nlargest(3,'ID')
    df1.reset_index(inplace=True)
    df2 = df.groupby(['base_office_name','Place_Of_Inspection']).count()[['ID']]
    df2.rename(columns={'ID':'Num_Offences'},inplace=True)
    df2.reset_index(inplace = True)
    df3=df2.groupby('base_office_name')
    df3=df3.apply(lambda x:x.sort_values(by=['Num_Offences'],ascending = False))
    df3 = df3.reset_index(drop=True)
    df4=df3.groupby('base_office_name').head(3)
    g3=df4[df4['base_office_name'].isin(df1['base_office_name'])]
    
    df2 = df.groupby(['region_name','Offence_name']).count()[['ID']]
    df2.rename(columns={'ID':'Num_Offences'},inplace=True)
    df2.reset_index(inplace=True)
    df3 = df2.groupby('region_name')
    df3 = df3.apply(lambda x:x.sort_values(by =['Num_Offences'],ascending=False))
    df4 = df3.reset_index(drop=True)
    df4 = df4.groupby('region_name').head(3)
    g4 = df4[df4['region_name'].isin(df['region_name'])]
        
    df3 = df.groupby(['Reg_Region','Offence_name']).count()[['ID']]
    df3.rename(columns={'ID':'Num_Offences'},inplace=True)
    df3.reset_index(inplace=True)
    df4 = df3.groupby('Reg_Region')
    df3 = df4.apply(lambda x:x.sort_values(by =['Num_Offences'],ascending=False))
    df4 = df3.reset_index(drop=True)
    df4 = df4.groupby('Reg_Region').head(3)
    g5 = df4[df4['Reg_Region'].isin(df['Reg_Region'])] 
        
    
    return[g2,g3,g4,g5]


# In[6]:


def drawfiggeo(entered_year):
    [g2,g3,g4,g5]=geodata(entered_year)
    if(entered_year=='2018-2021'):
        fig1 = px.treemap(g2, path=['region_name','Reg_Region'],values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='COMPARISON BETWEEN INSPECTION AND REGISTERED REGION')
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig2=px.sunburst(g3, path=[ 'base_office_name','Place_Of_Inspection'], values='Num_Offences',title='BASE OFFICES AND PLACES OF INSPECTION WHERE HIGHEST NUMBER OFFENCES OCCURED')
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig3 = px.treemap(g4, path=[ 'region_name','Offence_name'], values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='MAJOR OFFENCES BY REGION(INSPECTION)')
        fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig4 = px.treemap(g5, path=['Reg_Region','Offence_name'], values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='MAJOR OFFENCES BY VEHICEL REGISTERED IN DIFFERENT REGION')
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    else:
        fig1 = px.treemap(g2, path=['region_name','Reg_Region'],values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='COMPARISON BETWEEN INSPECTION AND REGISTERED REGION')
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig2=px.sunburst(g3, path=[ 'base_office_name','Place_Of_Inspection'], values='Num_Offences',title='TOP THREE OFFENCES BY BASE OFFICE AND PLACES OF INSPECTION')
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig3 = px.treemap(g4, path=[ 'region_name','Offence_name'], values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='MAJOR OFFENCES BY REGION(INSPECTION)')
        fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig4 = px.treemap(g5, path=['Reg_Region','Offence_name'], values='Num_Offences',color='Num_Offences',color_continuous_scale='RdBu',title='MAJOR OFFENCES BY VEHICEL REGISTERED IN DIFFERENT REGION')
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
    return[fig1,fig2,fig3,fig4]


# In[7]:


def demodata(entered_year):
    df_offences = pd.read_csv('rinchen6.csv')
    df_offences.drop(['Unnamed: 0','Driving_Region'],axis=1,inplace=True)
    
    if(entered_year=='2018-2021'):
        df = df_offences.copy()
    else:
        df = df_offences[df_offences['year']==int(entered_year)].copy()
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84,89])
    f1 = pd.crosstab(df.Age_group, df.Gender)
    f1.reset_index(inplace=True)
    f1[['Age_group']] = f1[['Age_group']].astype(str)
        
    f2 = pd.crosstab(df.Age_group, df.year)
    f2.reset_index(inplace=True)
    f2[['Age_group']] = f2[['Age_group']].astype(str)

    f3 = df.groupby(['Gender','Offence_name']).count()[['ID']]
    f3.reset_index(inplace = True)
    f3.rename(columns = {'ID':'Num_Offence'},inplace = True)
    f3 = f3.groupby('Gender')
    f3 = f3.apply(lambda x:x.sort_values(by = ['Num_Offence'], ascending = False))
    f3 = f3.reset_index(drop = True)
    f3 = f3.groupby('Gender').head(5)

    f4 = df.groupby(['Gender','year']).count()[['Offence_name']]
    f4.reset_index(inplace = True)
    f4.rename(columns = {'Offence_name':'Num_Offence'},inplace = True)
      
    df['ownership_type'] = df['ownership_type'].str.upper()
    f5 = df.groupby(['ownership_type','year']).count()[['ID']]
    f5.reset_index(inplace = True)
    f5.rename(columns = {'ID':'Num_Offence'}, inplace = True)
    
    f6 = df.groupby(['ownership_type','Offence_name']).count()[['ID']]
    f6.reset_index(inplace = True)
    f6.rename(columns = {'ID':'Num_Offence'},inplace = True)
    f6 = f6.groupby('ownership_type')
    f6 = f6.apply(lambda x:x.sort_values(by = ['Num_Offence'], ascending = False))
    f6 = f6.reset_index(drop = True)
    f6 = f6.groupby('ownership_type').head(3)
    
    f7 = df.groupby(['Vehicle_Type_Name']).count()[['Offence_name']]
    f7.reset_index(inplace = True)
    f7.rename(columns = {'Offence_name':'Num_Offences'})
    
    ff1 = df.groupby(['Vehicle_Type_Name','Offence_name']).count()[['ID']].nlargest(80,'ID')
    ff1.reset_index(inplace = True)

    ff2 = df.groupby(['Vehicle_Type_Name','Offence_name']).count()[['ID']]
    ff2.rename(columns = {'ID':'Num_Offence'},inplace = True)
    ff2.reset_index(inplace = True)

    ff2 = ff2.groupby('Vehicle_Type_Name')
    ff2 = ff2.apply(lambda x:x.sort_values(by = ['Num_Offence'], ascending = False))
    ff2 = ff2.reset_index(drop = True)
    ff2 = ff2.groupby('Vehicle_Type_Name').head(3)
    f8 = ff2[ff2['Vehicle_Type_Name'].isin(ff1['Vehicle_Type_Name'])]
 
    return[f1,f2,f3,f4,f5,f6,f7,f8]


# In[8]:


def drawfigdemo(entered_year):
    [f1,f2,f3,f4,f5,f6,f7,f8]=demodata(entered_year)
    if(entered_year=='2018-2021'):
        y = f2['Age_group']
        x1 = f2[2018]
        x2 = f2[2021]*-1
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(y=y,x=x1,name='2018',text = f2[2018],hoverinfo='text',orientation='h'))
        fig1.add_trace(go.Bar(y=y,x=x2,name='2021',text = f2[2021],hoverinfo='text',orientation='h'))
        fig1.update_layout(
        autosize = False,
        width = 1000,
        height=600,
        hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"),
        template = 'plotly_white',
        title = 'Number of offences for 2018 and 2021',
        barmode='relative',
        bargap=0.0,
        bargroupgap=0,
        xaxis=dict(
            tickvals=[-4000,-2000,0,2000,4000],
            ticktext=['4000','2000','0','2000','4000'],
            title='Number of Offences'))
        
        fig2 = px.bar(f4, x="year", y="Num_Offence", color="Gender",color_discrete_sequence=['pink','blue'], title="Gender base on Number of offences",width=700,height=600,
             labels={'Num_Offence':'Total Number of Offences'})
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig3 = px.bar(f5, x="year", y="Num_Offence",color='ownership_type',title="Distributions by onwership type over the time",barmode='stack',color_discrete_map={
        'BT': 'yellow',
        'BP': 'red',
        'BG':'blue'},labels={'Num_Offence':'Number of Offence'})
        fig3.update_layout(autosize = False,
                 width = 1000,
                 height=600,
                 margin = dict(t=30, l=25, r=25, b=50),
                 xaxis = dict(tickvals = list(np.arange(2018,2022)),
                 ticktext = ['2018','2019','2020','2021'],
                 title_font_size = 14),
                 legend=dict(yanchor="top", y=0.99, xanchor="right",x=0.99))
        fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        
        fig4 = px.pie(f7, values='Offence_name', names='Vehicle_Type_Name',title='Distributions by Vehicle type')
        fig4.update_traces(textposition='outside',textinfo='percent + label')
        fig4.update_layout(autosize = False,
                         width = 800,
                         height=600,
                         margin = dict(l=50,r=50,b=100,t=100,pad=4))
        fig4.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
                           
    else:
        y = f1['Age_group']
        x1 = f1['M']
        x2 = f1['F']*-1
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(y=y,x=x1,name='Male',text = f1['M'],hoverinfo='text',orientation='h'))
        fig1.add_trace(go.Bar(y=y,x=x2,name='Female',text = f1['F'],hoverinfo='text',orientation='h'))
        fig1.update_layout(
        autosize = False,
        width = 1000,
        height=600,
        hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"),
        template = 'plotly_white',
        title = 'Number of offences by Male and Female',
        barmode='relative',
        bargap=0.0,
        bargroupgap=0,
        xaxis=dict(
            tickvals=[-4000,-2000,0,2000,4000],
            ticktext=['4000','2000','0','2000','4000'],
            title='Number of Offences'))
            
        fig2 = px.bar(f3, x="Num_Offence", y="Offence_name", color="Gender",color_discrete_sequence=['pink','blue'],barmode="stack", title="Gender base on Number of offences",width=700,height=600,
             labels={'Offence_name':'Offences','Num_Offence':'Total Number of Offences'})
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"),
                         autosize = False,
                         width = 1000,
                         height=600)
        
        fig3 = px.bar(f6, x="Num_Offence", y="Offence_name",color='ownership_type',title="Major Offences by ownership type over time",barmode='stack',color_discrete_map={
        'BT': 'yellow',
        'BP': 'red',
        'BG':'blue'},labels={'Num_Offence':'Number of Offences','Offence_name':'Offences'})
        fig3.update_layout(autosize = False,
                         width = 1000,
                         height=600,
                         hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"),
                         margin = dict(
                         l=50,
                         r=50,
                         b=100,
                         t=100,
                         pad=4))
   
        fig4 = px.sunburst(f8, path=['Vehicle_Type_Name','Offence_name'], values='Num_Offence',title='Major Offences by Vehicle type')
        fig4.update_layout(
            autosize=False,
            width=600,
            height=600,
            hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"),
            margin=dict(
                l=70,
                r=50,
                b=100,
                t=100,
                pad=4))      
    return[fig1,fig2,fig3,fig4]


# In[9]:


# Create a dash application
app = JupyterDash(__name__)
app.title="Traffic Offence"
server=app.server


# In[10]:


tab_sty={
    'background':'#FFD700',
    'text-transform':'uppercase',
    'color':'black',
    'border':'grey',
    'font-size':'20px',
    'font-weight':400,
    'align-items':'center',
    'justifi-content':'center',
    'border-radius':'10px',
    'padding':'24px'
}
tab_selected_sty={
    'background':'#FFD700',
    'text-transform':'uppercase',
    'color':'blue',
    'font-size':'20px',
    'font-weight':1000,
    'align-items':'center',
    'justifi-content':'center',
    'border-radius':'10px',
    'padding':'24px'
}


# In[11]:


#html layout
year= ['2018','2019','2020','2021','2018-2021']

app.layout = html.Div([
    #wirte header
    html.H1('Welcome to Traffic Offence Report',style={'textAlign':'center','color':'#FC0A02','fontsize':50}),
    
    #1st division
    html.Div([
        dcc.Tabs(id="tabs",value="active",children=[
            dcc.Tab(label="Home",value="hom",style=tab_sty,selected_style=tab_selected_sty),
            dcc.Tab(label="Demographic",value="demo",style=tab_sty,selected_style=tab_selected_sty),
            dcc.Tab(label="Geographic",value="geo",style=tab_sty,selected_style=tab_selected_sty)
        ]),
    ],id="tabs_div"),
    
    html.Div([
        html.Div([
            html.H2('Select_Year',style={'textAlign':'left','color':'#9932CC','fontsize':40}),
            dcc.Dropdown(id='year_id',clearable=False,
                    options=[{'label':i,'value':i} for i in year ],
                    value='2018-2021'
                    ),
        ],id='dd',style={'width':'40%','padding':'3px','fontsize':40}),
        html.Div([
            html.H2('Number of Offences',style={'textAlign':'center','color':'#9932CC','fontsize':40}),
            html.Div(id="off_id",style={'height':35,'textAlign':'center','fontsize':40,
                                        'border-color':'cyan','background-color':'#00FF00','margin-left':'20px'}),
        ],id='no_box',style={'width':'40%','padding':'3px','fontsize':40}),
        
        html.Div([
            html.H2('Major Common Offence',style={'textAlign':'center','color':'#9932CC','fontsize':40}),
            html.Div(id="maj_id",style={'height':35,'textAlign':'center','fontsize':40,
                                        'border-color':'cyan','background-color':'#C0C0C0','margin-left':'20px'}),
        ],id='off_type',style={'width':'40%','padding':'3px','fontsize':40})
        
    ],id='dd_plus_value',style={'display':'flex'}),
    
    html.Br(),
    
    ##3rd Division                    
    html.Div([dcc.Graph(id='plot1'),
             dcc.Graph(id='plot2')
             ], style = {'display': 'flex'}),
    
    #4th division
    html.Div([
        # output graphic (plot2)
        dcc.Graph(id='plot3'),
        # output graphic (plot3)
        dcc.Graph(id='plot4')
        
    ], style = {'display': 'flex'}),             
])


# In[12]:


@app.callback([
    Output('off_id','children'),
    Output('maj_id','children'),
    Output('plot1','figure'),
    Output('plot2','figure'),
    Output('plot3','figure'),
    Output('plot4','figure')
],[Input('year_id','value'),
  Input('tabs','value')])

def draw_graph(entered_year,tabs): 
    if entered_year is None or tabs is None:
        raise PreventUpdate
    else:
        [op1,op2]=offencenum(entered_year)
        if (tabs=='hom'):
            [fig1,fig2,fig3,fig4]=drawfigtime(entered_year)
        elif(tabs=='geo'):
            [fig1,fig2,fig3,fig4]=drawfiggeo(entered_year)
        elif(tabs=='demo'):
            [fig1,fig2,fig3,fig4]=drawfigdemo(entered_year)
        else:
            [fig1,fig2,fig3,fig4]=drawfigtime(entered_year)

    return [op1,op2,fig1,fig2,fig3,fig4]
    


# In[13]:


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)    
    url = "http://127.0.0.1:{0}".format(port)    
    app.run_server(use_reloader=False, debug=True, port=port)

