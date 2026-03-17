import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardLayout from './components/layout/DashboardLayout';
import Overview from './pages/Overview';
import Portfolio from './pages/Portfolio';
import GnnInsights from './pages/GnnInsights';
import RlAgent from './pages/RlAgent';
import StressTesting from './pages/StressTesting';
import NasLab from './pages/NasLab';
import Federated from './pages/Federated';
import Quantum from './pages/Quantum';
import Sentiment from './pages/Sentiment';
import GraphVisualization from './pages/GraphVisualization';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route index element={<Overview />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="gnn" element={<GnnInsights />} />
          <Route path="rl" element={<RlAgent />} />
          <Route path="stress" element={<StressTesting />} />
          <Route path="nas" element={<NasLab />} />
          <Route path="fl" element={<Federated />} />
          <Route path="quantum" element={<Quantum />} />
          <Route path="sentiment" element={<Sentiment />} />
          <Route path="graph" element={<GraphVisualization />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
