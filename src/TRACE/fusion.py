class Fusion:
    """Class for fusing multiple predictions into a single result."""
    
    def __init__(self):
        pass
    
    def fuse(self, rule: str, **kwargs):
        """
        Fuse predictions according to the specified rule.
        
        Args:
            rule: The fusion rule to apply ('speakbench' or 's2sarena')
            **kwargs: Arguments specific to the fusion rule
            
        Returns:
            The fused prediction result
        """
        if rule == "speakbench":
            return self.speakbench_fusion(**kwargs)
        elif rule == "s2sarena":
            return self.s2sarena_fusion(**kwargs)
        else:
            raise NotImplementedError(f"Fusion rule '{rule}' is not implemented")
    
    def speakbench_fusion(self, content: str, para: str, vq: str) -> str:
        """
        Fuse predictions using the SpeakBench hierarchy.
        
        Hierarchy: Content > Instruction Following (para) > Voice Quality (vq)
        Includes the "Dominant both_bad" rule and handles missing data.
        
        Args:
            content: Content quality prediction
            para: Instruction following (paralinguistic) prediction
            vq: Voice quality prediction
            
        Returns:
            Fused prediction ('1', '2', 'both_good', or 'both_bad')
        """
        content = str(content)
        para = str(para)
        vq = str(vq)
        
        # Handle Missing Data
        if content == 'nan':
            return 'both_bad'
        
        # "Dominant both_bad" Rule
        if content == 'both_bad' and para == 'both_bad':
            return 'both_bad'
        
        # Original Hierarchy
        valid_winners = ['1', '2']
        if content in valid_winners:
            return content
        if para in valid_winners:
            return para
        if vq in valid_winners:
            return vq
        
        # Default Tie Handling
        if content == 'both_good':
            return 'both_good'
        
        return 'both_bad'
    
    def s2sarena_fusion(self, content: str, para: str, vq: str) -> str:
        """
        Fuse predictions using the S2SArena rule.
        
        Uses acceptability cap and typed ties with the RatingMin operator.
        Hierarchy: Content > Para > VQ, with acceptability capping from Content and Para.
        
        Args:
            content: Content quality prediction
            para: Instruction following (paralinguistic) prediction
            vq: Voice quality prediction
            
        Returns:
            Fused prediction ('1', '2', 'both_good', or 'both_bad')
        """
        content = str(content)
        para = str(para)
        vq = str(vq)
        
        # Calculate acceptability cap
        cap = self._rating_min(content, para)
        
        # Hierarchy with acceptability capping
        valid_winners = ['1', '2']
        
        if content in valid_winners:
            return self._rating_min(content, cap)
        
        if para in valid_winners:
            return self._rating_min(para, cap)
        
        if vq in valid_winners:
            return self._rating_min(vq, cap)
        
        # Default: use content with cap
        return self._rating_min(content, cap)
    
    def _rating_min(self, delta_a: str, delta_b: str) -> str:
        """
        RatingMin operator for S2SArena fusion.
        
        Computes element-wise minimum of binary acceptability ratings.
        
        Args:
            delta_a: First prediction ('1', '2', 'both_good', or 'both_bad')
            delta_b: Second prediction ('1', '2', 'both_good', or 'both_bad')
            
        Returns:
            Result of element-wise minimum ('1', '2', 'both_good', or 'both_bad')
        """
        # Convert predictions to binary acceptability tuples
        def to_tuple(delta: str) -> tuple:
            if delta == '1':
                return (1, 0)
            elif delta == '2':
                return (0, 1)
            elif delta == 'both_good':
                return (1, 1)
            elif delta == 'both_bad':
                return (0, 0)
            else:
                # Handle unexpected values by treating as both_bad
                return (0, 0)
        
        pi_a = to_tuple(delta_a)
        pi_b = to_tuple(delta_b)
        
        # Element-wise minimum
        pi_c = (min(pi_a[0], pi_b[0]), min(pi_a[1], pi_b[1]))
        
        # Convert back to prediction
        if pi_c == (1, 0):
            return '1'
        elif pi_c == (0, 1):
            return '2'
        elif pi_c == (1, 1):
            return 'both_good'
        else:  # (0, 0)
            return 'both_bad'