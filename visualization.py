import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set matplotlib backend to non-GTK before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import networkx as nx
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pyLDAvis
import pyLDAvis.sklearn
import streamlit.components.v1 as components
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Import our core utilities
from core_utils import format_date_uk, is_response_document

def plot_category_distribution(df: pd.DataFrame) -> None:
    """Plot category distribution"""
    all_cats = []
    for cats in df["categories"].dropna():
        if isinstance(cats, list):
            all_cats.extend(cats)

    cat_counts = pd.Series(all_cats).value_counts()

    fig = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        title="Category Distribution",
        labels={"x": "Category", "y": "Count"},
    )
    fig.update_layout(xaxis_title="Category", yaxis_title="Number of Reports", xaxis={"tickangle": 45})

    st.plotly_chart(fig, use_container_width=True)


def plot_coroner_areas(df: pd.DataFrame) -> None:
    """Plot coroner areas distribution"""
    area_counts = df["coroner_area"].value_counts().head(20)

    fig = px.bar(
        x=area_counts.index,
        y=area_counts.values,
        title="Top 20 Coroner Areas",
        labels={"x": "Area", "y": "Count"},
    )

    fig.update_layout(
        xaxis_title="Coroner Area",
        yaxis_title="Number of Reports",
        xaxis={"tickangle": 45},
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_timeline(df: pd.DataFrame) -> None:
    """Plot timeline of reports with improved formatting"""
    timeline_data = (
        df.groupby(pd.Grouper(key="date_of_report", freq="M")).size().reset_index()
    )
    timeline_data.columns = ["Date", "Count"]

    fig = px.line(
        timeline_data,
        x="Date",
        y="Count",
        title="Reports Timeline",
        labels={"Count": "Number of Reports"},
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        hovermode="x unified",
        yaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1,  # Integer steps
            rangemode="nonnegative",  # Ensure y-axis starts at 0 or above
        ),
        xaxis=dict(tickformat="%B %Y", tickangle=45),  # Month Year format
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_distribution(df: pd.DataFrame) -> None:
    """Plot monthly distribution with improved formatting"""
    # Format dates as Month Year
    df["month_year"] = df["date_of_report"].dt.strftime("%B %Y")
    monthly_counts = df["month_year"].value_counts().sort_index()

    fig = px.bar(
        x=monthly_counts.index,
        y=monthly_counts.values,
        labels={"x": "Month", "y": "Number of Reports"},
        title="Monthly Distribution of Reports",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Reports",
        xaxis_tickangle=45,
        yaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1,  # Integer steps
            rangemode="nonnegative",
        ),
        bargap=0.2,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_yearly_comparison(df: pd.DataFrame) -> None:
    """Plot year-over-year comparison with improved formatting"""
    yearly_counts = df["date_of_report"].dt.year.value_counts().sort_index()

    fig = px.line(
        x=yearly_counts.index.astype(int),  # Convert to integer years
        y=yearly_counts.values,
        markers=True,
        labels={"x": "Year", "y": "Number of Reports"},
        title="Year-over-Year Report Volumes",
    )

    # Calculate appropriate y-axis maximum
    max_count = yearly_counts.max()
    y_max = max_count + (1 if max_count < 10 else 2)  # Add some padding

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Reports",
        xaxis=dict(
            tickmode="linear",
            tick0=yearly_counts.index.min(),
            dtick=1,  # Show every year
            tickformat="d",  # Format as integer
        ),
        yaxis=dict(
            tickmode="linear", tick0=0, dtick=1, range=[0, y_max]  # Integer steps
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def analyze_data_quality(df: pd.DataFrame) -> None:
    """Analyze and display data quality metrics for PFD reports"""

    # Calculate completeness metrics
    total_records = len(df)

    def calculate_completeness(field):
        if field not in df.columns:
            return 0
        non_empty = df[field].notna()
        if field == "categories":
            non_empty = df[field].apply(lambda x: isinstance(x, list) and len(x) > 0)
        return (non_empty.sum() / total_records) * 100

    completeness_metrics = {
        "Title": calculate_completeness("Title"),
        "Content": calculate_completeness("Content"),
        "Date of Report": calculate_completeness("date_of_report"),
        "Deceased Name": calculate_completeness("deceased_name"),
        "Coroner Name": calculate_completeness("coroner_name"),
        "Coroner Area": calculate_completeness("coroner_area"),
        "Categories": calculate_completeness("categories"),
    }

    # Calculate consistency metrics
    consistency_metrics = {
        "Title Format": (df["Title"].str.len() >= 10).mean() * 100,
        "Content Length": (df["Content"].str.len() >= 100).mean() * 100,
        "Date Format": df["date_of_report"].notna().mean() * 100,
        "Categories Format": df["categories"]
        .apply(lambda x: isinstance(x, list))
        .mean()
        * 100,
    }

    # Calculate PDF metrics
    pdf_columns = [
        col for col in df.columns if col.startswith("PDF_") and col.endswith("_Path")
    ]
    reports_with_pdf = df[pdf_columns].notna().any(axis=1).sum()
    reports_with_multiple_pdfs = (df[pdf_columns].notna().sum(axis=1) > 1).sum()

    pdf_metrics = {
        "Reports with PDFs": (reports_with_pdf / total_records) * 100,
        "Reports with Multiple PDFs": (reports_with_multiple_pdfs / total_records)
        * 100,
    }

    # Display metrics using Streamlit
    st.subheader("Data Quality Analysis")

    # Completeness
    completeness_df = pd.DataFrame(
        list(completeness_metrics.items()), columns=["Field", "Completeness %"]
    )
    fig_completeness = px.bar(
        completeness_df,
        x="Field",
        y="Completeness %",
        title="Field Completeness Analysis",
    )
    fig_completeness.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_completeness, use_container_width=True)

    # Consistency
    consistency_df = pd.DataFrame(
        list(consistency_metrics.items()), columns=["Metric", "Consistency %"]
    )
    fig_consistency = px.bar(
        consistency_df, x="Metric", y="Consistency %", title="Data Consistency Analysis"
    )
    fig_consistency.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_consistency, use_container_width=True)

    # PDF Analysis
    pdf_df = pd.DataFrame(list(pdf_metrics.items()), columns=["Metric", "Percentage"])
    fig_pdf = px.bar(pdf_df, x="Metric", y="Percentage", title="PDF Coverage Analysis")
    fig_pdf.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pdf, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Completeness",
            f"{np.mean(list(completeness_metrics.values())):.1f}%",
        )

    with col2:
        st.metric(
            "Average Consistency", f"{np.mean(list(consistency_metrics.values())):.1f}%"
        )

    with col3:
        st.metric("PDF Coverage", f"{pdf_metrics['Reports with PDFs']:.1f}%")

    # Detailed quality issues
    st.markdown("### Detailed Quality Issues")

    issues = []

    # Check for missing crucial fields
    for field, completeness in completeness_metrics.items():
        if completeness < 95:  # Less than 95% complete
            issues.append(
                f"- {field} is {completeness:.1f}% complete ({total_records - int(completeness * total_records / 100)} records missing)"
            )

    # Check for consistency issues
    for metric, consistency in consistency_metrics.items():
        if consistency < 90:  # Less than 90% consistent
            issues.append(f"- {metric} shows {consistency:.1f}% consistency")

    # Check PDF coverage
    if pdf_metrics["Reports with PDFs"] < 90:
        issues.append(
            f"- {100 - pdf_metrics['Reports with PDFs']:.1f}% of reports lack PDF attachments"
        )

    if issues:
        st.markdown("The following quality issues were identified:")
        for issue in issues:
            st.markdown(issue)
    else:
        st.success("No significant quality issues found in the dataset.")


def display_topic_network(lda, feature_names):
    """Display word similarity network with interactive filters"""
    # st.markdown("### Word Similarity Network")
    st.markdown(
        "This network shows relationships between words based on their co-occurrence in documents."
    )

    # Store base network data in session state if not already present
    if "network_data" not in st.session_state:
        # Get word counts across all documents
        word_counts = lda.components_.sum(axis=0)
        top_word_indices = word_counts.argsort()[
            : -100 - 1 : -1
        ]  # Store more words initially

        # Create word co-occurrence matrix
        word_vectors = normalize(lda.components_.T[top_word_indices])
        word_similarities = cosine_similarity(word_vectors)

        st.session_state.network_data = {
            "word_counts": word_counts,
            "top_word_indices": top_word_indices,
            "word_similarities": word_similarities,
            "feature_names": feature_names,
        }

    # Network filters with keys to prevent rerun
    col1, col2, col3 = st.columns(3)
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Higher values show stronger connections only",
            key="network_min_similarity",
        )
    with col2:
        max_words = st.slider(
            "Number of Words",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Number of most frequent words to show",
            key="network_max_words",
        )
    with col3:
        min_connections = st.slider(
            "Minimum Connections",
            min_value=1,
            max_value=10,
            value=5,
            help="Minimum number of connections per word",
            key="network_min_connections",
        )

    # Create network graph based on current filters
    G = nx.Graph()

    # Get stored data
    word_counts = st.session_state.network_data["word_counts"]
    word_similarities = st.session_state.network_data["word_similarities"]
    top_word_indices = st.session_state.network_data["top_word_indices"][:max_words]
    feature_names = st.session_state.network_data["feature_names"]

    # Add nodes
    for idx, word_idx in enumerate(top_word_indices):
        G.add_node(idx, name=feature_names[word_idx], freq=float(word_counts[word_idx]))

    # Add edges based on current similarity threshold
    for i in range(len(top_word_indices)):
        for j in range(i + 1, len(top_word_indices)):
            similarity = word_similarities[i, j]
            if similarity > min_similarity:
                G.add_edge(i, j, weight=float(similarity))

    # Filter nodes by minimum connections
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) < min_connections:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    if len(G.nodes()) == 0:
        st.warning(
            "No nodes match the current filter criteria. Try adjusting the filters."
        )
        return

    # Create visualization
    pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

    # Create edge traces with varying thickness and color based on weight
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight * 3, color=f"rgba(100,100,100,{weight})"),
            hoverinfo="none",
            mode="lines",
        )
        edge_traces.append(edge_trace)

    # Create node trace with size based on frequency
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        freq = G.nodes[node]["freq"]
        name = G.nodes[node]["name"]
        connections = G.degree(node)
        node_text.append(
            f"{name}<br>Frequency: {freq:.0f}<br>Connections: {connections}"
        )
        node_size.append(np.sqrt(freq) * 10)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_size, line=dict(width=1), color="lightblue", sizemode="area"
        ),
    )

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Word Network ({len(G.nodes())} words, {len(G.edges())} connections)",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add network statistics
    st.markdown("### Network Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Words", len(G.nodes()))
    with col2:
        st.metric("Number of Connections", len(G.edges()))
    with col3:
        if len(G.nodes()) > 0:
            density = 2 * len(G.edges()) / (len(G.nodes()) * (len(G.nodes()) - 1))
            st.metric("Network Density", f"{density:.2%}")


def get_top_words(model, feature_names, topic_idx, n_words=10):
    """Get top words for a given topic"""
    return [
        feature_names[i]
        for i in model.components_[topic_idx].argsort()[: -n_words - 1 : -1]
    ]


def render_framework_heatmap(filtered_df, top_n_themes=5):
    """
    Create a framework-based heatmap of theme distribution by year with framework coloring
    
    Args:
        filtered_df: Filtered DataFrame containing theme analysis results
        top_n_themes: Number of top themes to show per framework
        
    Returns:
        Plotly figure object
    """
    if "year" not in filtered_df.columns or filtered_df["year"].isna().all():
        return None
    
    # Create combined framework:theme field
    filtered_df['Framework_Theme'] = filtered_df['Framework'] + ': ' + filtered_df['Theme']
    
    # Count reports per year (for denominator)
    # Assuming Record ID is the unique identifier for reports
    id_column = 'Record ID' if 'Record ID' in filtered_df.columns else filtered_df.columns[0]
    reports_per_year = filtered_df.groupby('year')[id_column].nunique()
    
    # Count unique report IDs per theme per year
    counts = filtered_df.groupby(['year', 'Framework', 'Framework_Theme'])[id_column].nunique().reset_index()
    counts.columns = ['year', 'Framework', 'Framework_Theme', 'Count']
    
    # Calculate percentages
    counts['Total'] = counts['year'].map(reports_per_year)
    counts['Percentage'] = (counts['Count'] / counts['Total'] * 100).round(1)
    
    # Get frameworks in the filtered data
    frameworks_present = filtered_df['Framework'].unique()
    
    # Get top themes by framework (N per framework)
    top_themes = []
    for framework in frameworks_present:
        framework_counts = counts[counts['Framework'] == framework]
        theme_totals = framework_counts.groupby('Framework_Theme')['Count'].sum().sort_values(ascending=False)
        top_themes.extend(theme_totals.head(top_n_themes).index.tolist())
    
    # Filter to top themes
    counts = counts[counts['Framework_Theme'].isin(top_themes)]
    
    # Create pivot table for heatmap
    pivot = counts.pivot_table(
        index='Framework_Theme',
        columns='year',
        values='Percentage',
        fill_value=0
    )
    
    # Create pivot for counts
    count_pivot = counts.pivot_table(
        index='Framework_Theme',
        columns='year',
        values='Count',
        fill_value=0
    )
    
    # Sort by framework then by total count
    theme_totals = counts.groupby('Framework_Theme')['Count'].sum()
    theme_frameworks = {theme: theme.split(':')[0] for theme in theme_totals.index}
    
    # Sort first by framework, then by count within framework
    sorted_themes = sorted(
        theme_totals.index,
        key=lambda x: (theme_frameworks[x], -theme_totals[x])
    )
    
    # Apply the sort order
    pivot = pivot.reindex(sorted_themes)
    count_pivot = count_pivot.reindex(sorted_themes)
    
    # Create color mapping for frameworks
    framework_colors = {}
    for i, framework in enumerate(frameworks_present):
        if framework == 'I-SIRch':
            framework_colors[framework] = "orange"  # Orange for I-SIRch
        elif framework == 'House of Commons':
            framework_colors[framework] = "royalblue"  # Blue for House of Commons
        else:
            # Use other colors for other frameworks
            other_colors = ["forestgreen", "purple", "darkred"]
            framework_colors[framework] = other_colors[i % len(other_colors)]
    
    # Create a visually distinctive dataframe for plotting
    # For each theme, create a dict with clean name and framework
    theme_display_data = []
    
    for theme in pivot.index:
        framework = theme.split(':')[0].strip()
        theme_name = theme.split(':', 1)[1].strip()
        
        # Insert line breaks for long theme names
        if len(theme_name) > 30:
            # Try to break at a space near the middle
            words = theme_name.split()
            if len(words) > 1:
                # Find a breaking point near the middle
                mid_point = len(words) // 2
                first_part = ' '.join(words[:mid_point])
                second_part = ' '.join(words[mid_point:])
                theme_name = f"{first_part}<br>{second_part}"
        
        theme_display_data.append({
            'original': theme,
            'clean_name': theme_name,
            'framework': framework,
            'color': framework_colors[framework]
        })
        
    theme_display_df = pd.DataFrame(theme_display_data)
    
    # Add year count labels
    year_labels = [f"{year}<br>n={reports_per_year[year]}" for year in pivot.columns]
    
    # Create heatmap using plotly
    fig = go.Figure()
    
    # Add heatmap
    heatmap = go.Heatmap(
        z=pivot.values,
        x=year_labels,
        y=theme_display_df['clean_name'],
        colorscale='Blues',
        zmin=0,
        zmax=min(100, pivot.values.max() * 1.2),  # Cap at 100% or 20% higher than max
        colorbar=dict(title='Percentage (%)'),
        hoverongaps=False,
        text=count_pivot.values,  # This will show the count in the hover
        hovertemplate='Year: %{x}<br>Theme: %{y}<br>Percentage: %{z}%<br>Count: %{text}<extra></extra>'
    )
    
    fig.add_trace(heatmap)
    
    # Add count annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if pivot.iloc[i, j] > 0:
                count = count_pivot.iloc[i, j]
                
                # Add an annotation for the actual count
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"({count})",
                    font=dict(size=8, color='black' if pivot.iloc[i, j] < 50 else 'white'),
                    showarrow=False,
                    xanchor='center',
                    yanchor='top',
                    yshift=-12
                )
    
    # Set y-axis ordering and color-coding
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(theme_display_df))),
            ticktext=theme_display_df['clean_name'],
            tickfont=dict(
                size=11,
                color='black'
            ),
        )
    )
    
    # Add colored framework indicators
    for framework, color in framework_colors.items():
        # Count themes for this framework
        framework_theme_count = theme_display_df[theme_display_df['framework'] == framework].shape[0]
        
        if framework_theme_count > 0:
            # Add a shape for the framework indicator
            fig.add_shape(
                type="rect",
                x0=-1,  # Slightly to the left of the y-axis
                x1=-0.5,
                y0=len(theme_display_df) - theme_display_df[theme_display_df['framework'] == framework].index[0] - framework_theme_count,
                y1=len(theme_display_df) - theme_display_df[theme_display_df['framework'] == framework].index[0],
                fillcolor=color,
                opacity=0.6,
                layer="below",
                line=dict(width=0)
            )
    
    # Add framework legend
    for i, (framework, color) in enumerate(framework_colors.items()):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=framework,
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Theme Distribution by Year",
        xaxis_title="Year (number of reports)",
        yaxis_title="Theme",
        height=len(pivot.index) * 30 + 200,  # Adjust height based on number of themes
        margin=dict(l=200, r=20, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_lda_visualization(lda_model, vectorizer, documents):
    """Create interactive LDA visualization using pyLDAvis"""
    try:
        # Prepare the LDA visualization
        vis_data = pyLDAvis.sklearn.prepare(
            lda_model, 
            vectorizer.transform(documents), 
            vectorizer,
            mds='tsne',  # Use t-SNE for better separation
            sort_topics=False
        )
        
        # Save to temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            pyLDAvis.save_html(vis_data, f.name)
            
            # Read the HTML content
            with open(f.name, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()
            
            # Display in Streamlit
            components.html(html_content, height=800, scrolling=True)
            
    except Exception as e:
        st.error(f"Error creating LDA visualization: {str(e)}")
        logging.error(f"LDA visualization error: {e}", exc_info=True)


def plot_theme_confidence_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of theme confidence scores"""
    if "Combined Score" not in df.columns:
        st.warning("No confidence scores found in the data")
        return
    
    fig = px.histogram(
        df,
        x="Combined Score",
        title="Distribution of Theme Confidence Scores",
        labels={"Combined Score": "Confidence Score", "count": "Number of Themes"},
        nbins=20
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Number of Themes",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_framework_comparison(df: pd.DataFrame) -> None:
    """Plot comparison between different frameworks"""
    if "Framework" not in df.columns:
        st.warning("No framework data found")
        return
    
    framework_counts = df["Framework"].value_counts()
    
    fig = px.pie(
        values=framework_counts.values,
        names=framework_counts.index,
        title="Distribution of Themes by Framework"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_themes_by_year(df: pd.DataFrame) -> None:
    """Plot themes distribution by year"""
    if "year" not in df.columns or "Theme" not in df.columns:
        st.warning("Missing year or theme data")
        return
    
    # Get top themes
    top_themes = df["Theme"].value_counts().head(10).index
    
    # Filter to top themes
    filtered_df = df[df["Theme"].isin(top_themes)]
    
    # Create yearly distribution
    yearly_themes = filtered_df.groupby(["year", "Theme"]).size().reset_index(name="Count")
    
    fig = px.line(
        yearly_themes,
        x="year",
        y="Count",
        color="Theme",
        title="Top Themes Distribution Over Time",
        labels={"year": "Year", "Count": "Number of Reports"}
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Reports",
        legend_title="Theme"
    )
    
    st.plotly_chart(fig, use_container_width=True) 


def improved_truncate_text(text, max_length=30):
    """
    Improved function to handle long text for chart display by breaking into lines
    instead of simple truncation with ellipses
    
    Args:
        text: String to format
        max_length: Maximum length per line
        
    Returns:
        Text with line breaks inserted at appropriate word boundaries
    """
    if not text or len(text) <= max_length:
        return text
    
    # For theme names with ":" in them (framework:theme format)
    if ":" in text:
        parts = text.split(":", 1)
        framework = parts[0].strip()
        theme = parts[1].strip()
        
        # For theme part, break it into lines rather than truncate
        if len(theme) > max_length - len(framework) - 2:  # -2 for ": "
            # Process the theme part with word-aware line breaking
            words = theme.split()
            processed_theme = []
            current_line = []
            current_length = 0
            
            for word in words:
                # If adding this word keeps us under the limit
                if current_length + len(word) + (1 if current_line else 0) <= max_length - len(framework) - 2:
                    current_line.append(word)
                    current_length += len(word) + (1 if current_line else 0)
                else:
                    # Line is full, start a new one
                    processed_theme.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            # Add the final line if any
            if current_line:
                processed_theme.append(" ".join(current_line))
            
            # If we have more than 2 lines, keep first 2 and add ellipsis
            if len(processed_theme) > 2:
                return f"{framework}: {processed_theme[0]}<br>{processed_theme[1]}..."
            else:
                # Join with line breaks
                return f"{framework}: {('<br>').join(processed_theme)}"
        
        return f"{framework}: {theme}"
    
    # For normal long strings, add line breaks at word boundaries
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + (1 if current_line else 0) <= max_length:
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    # If we have more than 2 lines, keep first 2 and add ellipsis
    if len(lines) > 2:
        return f"{lines[0]}<br>{lines[1]}..."
    
    # For plotly charts, use <br> for line breaks
    return "<br>".join(lines)
  