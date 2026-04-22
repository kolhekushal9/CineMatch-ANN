const API_BASE = '/api';

const searchInp = document.getElementById('search-input');

// Initialize Dashboard immediately
function init() {
    loadRecommendations();
    loadMovies();
}

// Fetch Movies
document.getElementById('search-btn').addEventListener('click', () => {
    loadMovies(searchInp.value);
});

async function loadMovies(search = '') {
    const grid = document.getElementById('movies-grid');
    grid.innerHTML = '<div class="spinner"></div>';

    try {
        const res = await fetch(`${API_BASE}/movies?search=${search}`);
        const movies = await res.json();
        renderMovies(movies, grid);
    } catch(err) {
        grid.innerHTML = '<p>Error loading movies.</p>';
    }
}

// Fetch Recommendations
async function loadRecommendations() {
    const grid = document.getElementById('recommendations-grid');
    grid.innerHTML = '<div class="spinner"></div>';

    try {
        const res = await fetch(`${API_BASE}/recommendations`);
        const movies = await res.json();
        if(movies.length === 0 || movies.msg) {
            grid.innerHTML = '<p>No recommendations available yet. Rate some movies!</p>';
        } else {
            renderMovies(movies, grid, true);
        }
    } catch(err) {
        grid.innerHTML = '<p>Model still loading or error fetching recommendations.</p>';
    }
}

function renderMovies(movies, container, isRec = false) {
    container.innerHTML = '';
    movies.forEach(movie => {
        const card = document.createElement('div');
        card.className = `movie-card ${isRec ? 'recommendation' : ''}`;
        
        let predHtml = isRec && movie.predicted_rating 
            ? `<div class="pred-rating">AI Predicted Rating: ⭐ ${movie.predicted_rating}/5</div>` 
            : '';

        card.innerHTML = `
            <div class="movie-title">${movie.title}</div>
            <div class="movie-genres">${movie.genres.replace(/\|/g, ', ')}</div>
            ${predHtml}
            <div class="rating-stars" data-id="${movie.encoded}">
                <span class="star" data-val="1">★</span>
                <span class="star" data-val="2">★</span>
                <span class="star" data-val="3">★</span>
                <span class="star" data-val="4">★</span>
                <span class="star" data-val="5">★</span>
            </div>
        `;
        container.appendChild(card);
    });

    // Attach rating listeners
    document.querySelectorAll('.rating-stars').forEach(group => {
        const stars = group.querySelectorAll('.star');
        // Simple star hover logic (could be improved, but functional for UI)
        stars.forEach(s => {
            s.addEventListener('click', async (e) => {
                const val = e.target.getAttribute('data-val');
                const encodedId = group.getAttribute('data-id');
                // visual update
                stars.forEach(st => st.classList.remove('active'));
                for(let i=0; i<val; i++) stars[i].classList.add('active');
                
                // submit rating
                await rateMovie(encodedId, val);
            });
        });
    });
}

async function rateMovie(movie_encoded, rating) {
    try {
        await fetch(`${API_BASE}/ratings`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ movie_encoded, rating })
        });
        // Reload recommendations after rating
        loadRecommendations();
    } catch(err) {
        console.error("Failed to rate");
    }
}

init();

// Modal Logic
const modal = document.getElementById('ann-modal');
const btn = document.getElementById('how-it-works-btn');
const span = document.getElementById('close-modal');

if (btn && modal && span) {
    btn.onclick = function() {
        modal.classList.add('show');
    }
    span.onclick = function() {
        modal.classList.remove('show');
    }
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.classList.remove('show');
        }
    }
}
